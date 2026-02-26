# Audit: Process Supervision and Crash Recovery

**Date:** 2026-02-25
**Scope:** Remote process management, crash recovery, health checking, checkpointing.
**Files examined:** `scripts/launch_remote.py`, `scripts/remote.py`, `scripts/remote_tmux.py`, `src_ii/training_resume.py`, `src_ii/incremental_save.py`, `src_ii/distributed.py`, `scripts/launch_server.py`, `scripts_ii/launch_server.py`, `docs/remote_deployment.md`, `docs/case_study_2xh100_20260216.md`

---

## Executive Summary

There is **no process supervision** on the remote machine. Training and inference processes run inside tmux sessions with zero automatic restart, zero health monitoring, and zero crash recovery. If a process crashes, it is simply dead until a human notices and manually restarts it. The `ServerCluster` class in `launch_remote.py` (which runs locally and manages local subprocess servers) has graceful SIGTERM-based shutdown, but there is no equivalent for remote processes. The training resume infrastructure (`training_resume.py`) provides checkpoint-based recovery for the optimizer state, but it requires manual invocation -- there is nothing that detects a crashed training run and resumes it. The `distributed.py` module provides NCCL-based gradient sync but is **completely unused** in all training scripts.

**Critical Issues: 3 | Major Issues: 3 | Minor Issues: 2**

---

## Detailed Findings

### 1. No process supervisor whatsoever (Critical)

There is no systemd unit, no supervisord config, no watchdog, no process manager of any kind. Remote processes are launched via:

> ```python
> # scripts/remote.py _remote_tmux_launch(), lines 198-204
> script = (
>     f"mkdir -p ~/.futudiffu_logs && "
>     f"tmux kill-session -t {full_name} 2>/dev/null; "
>     f"tmux new-session -d -s {full_name} '{command}' && "
>     f"tmux pipe-pane -t {full_name} -o 'cat >> {log_path}'"
> )
> ```

When the command inside the tmux session exits (whether gracefully or via crash), the tmux session itself exits. The session is gone. There is no mechanism to:
- Detect that a process crashed
- Automatically restart it
- Notify anyone that it stopped
- Log the exit code

The tmux session simply disappears from `tmux list-sessions`, and the only evidence is the pipe-pane log file (which may not have captured the final output, as discussed in the tmux audit).

### 2. No health checking or heartbeat (Critical)

There is no periodic health check for any remote process. The `_wait_for_remote_port()` function (lines 516-535 in `remote.py`) polls a port during initial server launch:

> ```python
> def _wait_for_remote_port(config, port, timeout=180.0, poll_interval=3.0):
>     deadline = time.monotonic() + timeout
>     while time.monotonic() < deadline:
>         result = _ssh(config,
>             f"python3 -c \"import socket; s=socket.socket(); "
>             f"s.settimeout(1); s.connect(('127.0.0.1', {port})); "
>             f"s.close(); print('open')\" 2>/dev/null || echo closed",
>             check=False, timeout=15)
>         if "open" in result.stdout:
>             return True
>         time.sleep(poll_interval)
>     return False
> ```

This is only called once during `cmd_launch --wait`. After that, nothing monitors whether the server is still alive. If the inference server crashes 10 minutes into a training run, the training script will get connection errors on its next RPC call, but there is nothing to detect and report the server death independently.

The legacy `scripts/launch_server.py` has a `--heartbeat-file` argument:

> ```python
> "--heartbeat-file", r"F:\dox\repos\ai\futudiffu\server_heartbeat.bin",
> ```

But this is a Windows-only local configuration. The heartbeat mechanism is in the ZMQ server (`src/futudiffu/server.py`), which is frozen/dead architecture. The new FastAPI server (`src_ii/server.py`) has no heartbeat mechanism.

### 3. VM crash does not trigger automatic recovery (Critical)

The documentation in `docs/remote_deployment.md` explicitly acknowledges this:

> ```
> What does NOT survive:
> - Server processes (must relaunch)
> - torch.compile caches (will rewarm, ~1-2 min)
> - In-flight training state (LoRA weights are in server VRAM)
> - Dataset v2 WIP blobs (buffers sealed only on writer close)
> ```

If the VM is preempted or crashes:
1. All tmux sessions are lost (tmux runs in userspace, not persisted across reboots)
2. All GPU memory (including in-training LoRA weights) is lost
3. The model must be reloaded from disk
4. torch.compile must re-warm

The resume path documented is fully manual:

> ```
> Resume strategy:
> # Re-run the same command. Skip-if-present handles models.
> python scripts/launch_remote.py --model-dir ./models --n-gpus 2
> # After validation passes, resume training from last checkpoint:
> python scripts/train.py ... --resume ...
> ```

There is no cron job, no cloud-init script, no systemd service that would automatically restart training after a VM reboot.

### 4. Training checkpoint resume is manual-only (Major)

`src_ii/training_resume.py` provides `detect_resume()` which scans for the latest checkpoint:

> ```python
> def detect_resume(output_dir):
>     # Scans for checkpoint_stepNNN/ directories with resume_state.json
>     # Returns {checkpoint_dir, step, rng_state, training_curve_len}
> ```

This is called when a user passes `--resume` to the training script. It correctly finds the latest checkpoint and loads optimizer state. However:

- There is no automatic invocation. A crashed training run must be manually restarted with `--resume`.
- The checkpoint interval is configurable but defaults are not aggressive. The case study recommends `--checkpoint-every 5` for spot instances, but nothing enforces this.
- There is no verification that the checkpoint is complete (e.g., all files present and not truncated).

### 5. Signal handling exists locally but not remotely (Major)

`launch_remote.py` has a SIGINT handler for the local orchestrator:

> ```python
> # scripts/launch_remote.py lines 43-53
> _interrupted = False
> def _signal_handler(signum, frame):
>     global _interrupted
>     if _interrupted:
>         print("\nForce quit.", flush=True)
>         sys.exit(1)
>     _interrupted = True
>     print("\nInterrupt received. Skipping remaining phases...", flush=True)
> ```

And the `ServerCluster` class sends SIGTERM to child processes on exit:

> ```python
> # lines 218-229
> def __exit__(self, *exc):
>     for i, proc in enumerate(self._procs):
>         if proc.poll() is None:
>             proc.send_signal(signal.SIGTERM)
>     for proc in self._procs:
>         try:
>             proc.wait(timeout=10)
>         except subprocess.TimeoutExpired:
>             proc.kill()
> ```

But this only applies to processes managed as local subprocesses. Remote tmux sessions have no signal handling. When `_remote_tmux_kill()` is called, it sends `tmux kill-session`, which sends SIGHUP to all processes in the session. The inference server and training scripts do not register SIGHUP handlers, so they may not flush buffers or write final checkpoints before dying.

### 6. distributed.py is completely unused (Major)

`src_ii/distributed.py` provides `setup_distributed()`, `make_gradient_sync_fn()`, `is_rank_zero()`, `barrier()`, and `cleanup()`. These are designed for multi-GPU data-parallel training via NCCL.

However, searching the entire codebase reveals that **no training script imports or calls any of these functions**. The multi-GPU strategy actually used (documented in the case study) is independent inference servers per GPU with a client-side orchestrator -- not NCCL-based data parallelism.

`btrm_training.py` accepts a `gradient_sync_fn` parameter, but no caller provides one. All training runs effectively single-GPU.

### 7. No PID file or lock file mechanism (Minor)

There are no PID files anywhere in the codebase. This means:
- There is no way to detect stale processes from a previous run
- Two copies of the same server can be started on the same port (the second will fail to bind, but the error message won't indicate a running duplicate)
- There is no lock to prevent concurrent training runs writing to the same output directory

### 8. GPU memory is not explicitly freed on crash (Minor)

If a training or inference process crashes without cleanup, GPU memory is released by the CUDA driver when the process exits. This is correct behavior at the process level. However, if the process is a Python process that segfaults in a C extension (e.g., NCCL, Triton), the CUDA context may not be properly destroyed, leaving "zombie" GPU memory allocations that persist until the GPU is reset.

The case study notes that server restarts were clean because the entire process was killed:

> ```
> model_manager.py handles all VRAM allocation... There is no "stale CUDA context"
> or "leaked GPU memory" because the process exits cleanly and the new process
> starts from a blank VRAM state.
> ```

This relies on the process actually exiting (which it does if killed by tmux session teardown). But there is no `nvidia-smi` verification after crash recovery.

---

## Summary Table

| # | Issue | Severity |
|---|-------|----------|
| 1 | No process supervisor (systemd, supervisord, etc.) | Critical |
| 2 | No health checking or heartbeat monitoring | Critical |
| 3 | VM crash/preemption has no automatic recovery | Critical |
| 4 | Training checkpoint resume is manual-only | Major |
| 5 | Signal handling local only, not remote | Major |
| 6 | distributed.py is completely unused | Major |
| 7 | No PID files or lock files | Minor |
| 8 | No GPU memory verification after crash | Minor |

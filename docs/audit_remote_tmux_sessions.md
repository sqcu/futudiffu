# Audit: Remote tmux Session Persistence and Lifecycle

**Date:** 2026-02-25
**Scope:** All tmux session creation, attachment, and lifecycle code in the repository.
**Files examined:** `scripts/remote.py`, `scripts/remote_tmux.py`, `docs/remote_sessions.md`, `docs/case_study_2xh100_20260216.md`, `docs/live_run_defects_20260216.md`

---

## Executive Summary

The tmux infrastructure is split across two scripts with fundamentally different designs: `remote_tmux.py` (runs ON the remote machine, has has-session checking, refuses to clobber existing sessions) and `remote.py` (runs LOCALLY, dispatches tmux commands via SSH, unconditionally kills existing sessions before creating new ones). These two scripts are mutually unaware and use overlapping session name spaces. The remote-side script is better designed but is not integrated into the remote orchestration workflow. The SSH-dispatched tmux in `remote.py` has no reconnection awareness, no environment setup within sessions, and uses a kill-then-create pattern that destroys running work if invoked carelessly.

**Critical Issues: 2 | Major Issues: 4 | Minor Issues: 3**

---

## Detailed Findings

### 1. Two competing tmux management systems (Major)

There are two separate tmux management scripts:

- **`scripts/remote_tmux.py`** (287 lines) -- Designed to run ON the remote machine. Has `_session_exists()` checking via `tmux has-session`. Refuses to clobber existing sessions (`"warning: session already exists, not clobbering"`). Has `launch`, `attach`, `watch` (read-only), `list`, `logs`, `kill`, and a `setup` command for the server/train/sync trio.

- **`scripts/remote.py`** (803 lines) -- Designed to run locally, dispatching commands via SSH. Its `_remote_tmux_launch()` function unconditionally kills any existing session with the same name before creating a new one:

> ```python
> # scripts/remote.py lines 198-204
> script = (
>     f"mkdir -p ~/.futudiffu_logs && "
>     f"tmux kill-session -t {full_name} 2>/dev/null; "
>     f"tmux new-session -d -s {full_name} '{command}' && "
>     f"tmux pipe-pane -t {full_name} -o 'cat >> {log_path}'"
> )
> ```

The `;` after `kill-session` means "continue regardless of exit code." This is a destructive-by-default pattern. If `cmd_launch` is called a second time (e.g., after reconnecting), it silently destroys the running servers and starts new ones.

`remote_tmux.py` is never invoked by `remote.py`. They are independent tools that happen to use the same `fd_` prefix convention.

### 2. No session reconnection logic in remote.py (Critical)

When a user's SSH session drops and they reconnect, there is no "reconnect to existing sessions" workflow in `remote.py`. The `cmd_status` command lists sessions, and `cmd_logs` reads log files, but there is no mechanism to:

- Detect that sessions are still running from a prior connection
- Resume monitoring without restarting
- Gracefully transition from "cold reconnect" to "active monitoring"

The workflow after a network drop is entirely manual: run `status` to see what's alive, then `logs` to tail each one. There's no single command that says "show me the state of my running session."

### 3. Unconditional kill-before-create destroys running work (Critical)

`_remote_tmux_launch()` (line 198-204 in `remote.py`) kills any session with the matching name before creating a new one. This is the ONLY session creation path in `remote.py`. Every command that launches a tmux session (`cmd_validate`, `cmd_launch`, `cmd_upload`, `cmd_models --hf`) goes through this function.

If a user accidentally re-runs `remote.py launch` while servers are actively serving training requests, the running servers are killed without warning. The training process (which is a separate tmux session) will then fail on its next RPC call with no clear error pointing to "your server was just killed."

`remote_tmux.py` correctly handles this by checking `_session_exists()` and refusing to clobber. But `remote_tmux.py` is not used by `remote.py`.

### 4. No environment setup within tmux sessions (Major)

tmux sessions created by `remote.py` launch commands directly without activating the venv or setting up the environment. The command string manually includes `PYTHONPATH` and `CUDA_VISIBLE_DEVICES`:

> ```python
> # scripts/remote.py lines 488-496 (cmd_launch)
> cmd = (
>     f"cd {config.remote_dir} && "
>     f"CUDA_VISIBLE_DEVICES={i} "
>     f"PYTHONPATH={config.remote_dir}/src "
>     f"{python} -m futudiffu.server "
>     ...
> )
> ```

There is no `.bashrc` sourcing, no venv activation, no `LD_LIBRARY_PATH` setup for CUDA libraries. If the system Python's environment differs from the venv (which is typical on cloud VMs), the process may pick up wrong library versions.

`remote_tmux.py` also does not set up environment within sessions -- it passes the raw command string to `tmux new-session -d -s SESSION COMMAND`.

### 5. pipe-pane logging is set up after session creation (Major)

Both scripts create the tmux session and THEN set up pipe-pane logging as a separate command:

> ```python
> # scripts/remote.py lines 201-204
> f"tmux new-session -d -s {full_name} '{command}' && "
> f"tmux pipe-pane -t {full_name} -o 'cat >> {log_path}'"
> ```

> ```python
> # scripts/remote_tmux.py lines 95-101
> subprocess.run(
>     ["tmux", "new-session", "-d", "-s", session, command],
>     check=True,
> )
> subprocess.run(
>     ["tmux", "pipe-pane", "-t", session, "-o", f"cat >> {log}"],
>     check=True,
> )
> ```

In `remote.py`, both commands are in a single SSH command joined by `&&`, so if the `new-session` succeeds but the `pipe-pane` setup fails (e.g., tmux version incompatibility), the session runs without logging. In `remote_tmux.py`, they are separate subprocess calls, so output between session creation and pipe-pane setup is lost. For fast-failing processes, this can mean the error message is never captured.

### 6. No tmux session cleanup on teardown failure (Major)

`cmd_teardown` in `remote.py` (lines 673-698) lists sessions and kills them one by one. If a `kill-session` fails:

> ```python
> for name, _ in sessions:
>     try:
>         _remote_tmux_kill(config, name)
>     except subprocess.CalledProcessError:
>         print(f"  warning: failed to kill '{name}'")
> ```

The warning is printed but teardown continues. There is no retry, no verification that the session actually died, and no escalation (e.g., `kill -9` on the tmux server or the processes within the session).

### 7. Log file accumulation without rotation (Minor)

Logs are written to `~/.futudiffu_logs/{session_name}.log` with `cat >>` (append mode). There is no log rotation, no size limits, and no cleanup. Over multiple sessions, logs from previous runs accumulate indefinitely. The log path is deterministic by session name, so a new session named `server_0` appends to the same log file as a previous `server_0` session, making it impossible to distinguish log entries from different runs without timestamps.

### 8. Session names not globally unique (Minor)

Session names like `server_0`, `server_1`, `validate`, `model_download` are deterministic and reused across runs. Combined with the kill-before-create pattern in `remote.py` and the log append behavior, there is no isolation between runs. A production system would include a run ID or timestamp in session names.

### 9. No tmux configuration for scroll-back buffer (Minor)

Neither script configures tmux's history-limit. The default tmux scrollback buffer is 2000 lines. A long training run easily exceeds this. While pipe-pane captures output to a file, interactive `attach` or `watch` sessions only see the last 2000 lines. This is a usability issue, not a correctness issue.

---

## Summary Table

| # | Issue | Severity |
|---|-------|----------|
| 1 | Two competing tmux management systems | Major |
| 2 | No session reconnection logic in remote.py | Critical |
| 3 | Unconditional kill-before-create destroys running work | Critical |
| 4 | No environment setup within tmux sessions | Major |
| 5 | pipe-pane logging set up after session creation | Major |
| 6 | No cleanup verification on teardown failure | Major |
| 7 | Log file accumulation without rotation | Minor |
| 8 | Session names not globally unique | Minor |
| 9 | No tmux scrollback buffer configuration | Minor |

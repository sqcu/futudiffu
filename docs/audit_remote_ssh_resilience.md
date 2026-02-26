# Audit: SSH Connection Resilience and Reconnection

**Date:** 2026-02-25
**Scope:** All SSH connection code, transport primitives, and failure handling.
**Files examined:** `scripts/remote.py`, `scripts/remote_node_bootstrap.py`, `remote_target.json.example`, `docs/case_study_2xh100_20260216.md`, `docs/live_run_defects_20260216.md`

---

## Executive Summary

SSH connectivity is implemented entirely via subprocess calls to the system `ssh` and `rsync` binaries. There is no paramiko, no fabric, no SSH library whatsoever. While this avoids library-level complexity, it means there is zero programmatic control over the SSH connection -- no retry logic, no keepalive configuration, no connection pooling, and no handling of mid-command disconnects. The `StrictHostKeyChecking=accept-new` setting is a security concern for production use. SSH credentials are managed via a plaintext JSON config file pointing to a key file, with no permission validation. The case study documents SSH rate limiting on PrimeIntellect causing `Connection reset by peer` -- a known issue with zero mitigation in the code.

**Critical Issues: 2 | Major Issues: 3 | Minor Issues: 3**

---

## Detailed Findings

### 1. No SSH retry logic whatsoever (Critical)

The `_ssh()` function in `remote.py` (lines 132-153) executes a single `subprocess.run()` call with no retry:

> ```python
> def _ssh(
>     config: RemoteConfig, cmd: str,
>     timeout: int = 120, check: bool = True,
> ) -> subprocess.CompletedProcess:
>     full_cmd = _ssh_args(config) + [cmd]
>     result = subprocess.run(
>         full_cmd, capture_output=True, text=True, timeout=timeout,
>     )
>     if check and result.returncode != 0:
>         raise subprocess.CalledProcessError(...)
>     return result
> ```

If the SSH connection fails for any transient reason (network blip, DNS resolution delay, cloud provider throttling, TCP timeout), the entire operation fails immediately. There is no exponential backoff, no retry count, no distinction between transient and permanent failures. The case study documents that PrimeIntellect SSH rate limiting caused `Connection reset by peer` during the session (defect #17).

### 2. No SSH keepalive configuration (Critical)

The SSH arguments in `_ssh_args()` (lines 120-129) set `ConnectTimeout=10` and `BatchMode=yes` but no keepalive:

> ```python
> def _ssh_args(config: RemoteConfig) -> list[str]:
>     return [
>         "ssh",
>         "-i", config.ssh_key,
>         "-o", "StrictHostKeyChecking=accept-new",
>         "-o", "ConnectTimeout=10",
>         "-o", "BatchMode=yes",
>         config.host,
>     ]
> ```

Missing:
- `ServerAliveInterval` (send keepalive every N seconds to prevent NAT/firewall timeout)
- `ServerAliveCountMax` (how many missed keepalives before disconnecting)
- `TCPKeepAlive=yes` (OS-level TCP keepalive)

Without these, long-running SSH operations (file transfers, tailing logs with `--follow`) are vulnerable to silent connection drops from NAT gateways or cloud firewalls that time out idle TCP connections (typically 5-15 minutes).

### 3. StrictHostKeyChecking=accept-new auto-trusts new hosts (Major)

> ```python
> "-o", "StrictHostKeyChecking=accept-new",
> ```

This means the FIRST connection to any host is automatically trusted without verification. While `accept-new` is better than `no` (it will reject changed keys for known hosts), it provides zero protection against initial MITM attacks. For spot instances with dynamic IPs, this is a practical necessity, but it should be documented as a security trade-off.

### 4. rsync has no retry logic or progress monitoring (Major)

The `_rsync()` function (lines 156-182) runs rsync as a single subprocess call:

> ```python
> def _rsync(...) -> subprocess.CompletedProcess:
>     cmd = [
>         "rsync", "-avz", "--progress",
>         "-e", f"ssh -i {config.ssh_key} -o StrictHostKeyChecking=accept-new",
>     ]
>     ...
>     result = subprocess.run(cmd, timeout=timeout)
>     if result.returncode != 0:
>         raise subprocess.CalledProcessError(...)
>     return result
> ```

Problems:
- No `--partial` flag: if rsync is interrupted during a large file transfer, the partial file is deleted and the entire transfer restarts from zero.
- No `--partial-dir`: would allow resuming partial transfers.
- No retry on failure: a transient network issue during model file transfer (5.8 GB FP8 diff) requires re-running the entire command manually.
- The SSH options passed to rsync's `-e` flag do not include keepalive settings, so long transfers are vulnerable to the same NAT timeout issues as SSH.
- Timeout is set per-call (default 600s for general rsync, 3600s for model transfer), but a slow but progressing transfer will be killed at the timeout boundary.

### 5. SSH key permission not validated (Major)

The case study documents (defect #8):

> ```
> SSH key permissions not auto-fixed
> ~/.ssh/primeintellect_ed25519 had 0644, SSH refused it
> ```

`remote.py` reads `ssh_key` from `remote_target.json` and passes it directly to `-i`. There is no `chmod 600` or permission check before use. SSH silently refuses keys with permissive permissions, producing a generic "Permission denied" error that does not indicate the key file permissions are the cause.

### 6. No SSH connection pooling or multiplexing (Minor)

Each `_ssh()` call establishes a new SSH connection, performs the TCP handshake, key exchange, and authentication from scratch. The `_wait_for_session_exit()` function polls every 10 seconds via separate SSH connections:

> ```python
> def _wait_for_session_exit(...) -> bool:
>     while time.monotonic() < deadline:
>         result = _ssh(config, ...)
>         time.sleep(poll_interval)
> ```

For a 30-minute training session polled every 10 seconds, this is ~180 SSH connections just for monitoring. With PrimeIntellect's SSH rate limiting, this directly contributed to `Connection reset by peer` errors.

SSH `ControlMaster` and `ControlPath` would allow multiplexing all SSH connections over a single TCP connection, eliminating the rate limiting issue entirely.

### 7. Windows Python path expansion breaks SSH key resolution (Minor)

Defect #14 from the case study:

> ```
> remote.py run via .venv/Scripts/python.exe expanded ~ to Windows home
> Must run via /usr/bin/python3 for SSH key resolution
> ```

`remote_target.json.example` specifies `"ssh_key": "~/.ssh/your_key"`. The `RemoteConfig.load()` method calls `os.path.expanduser()` (line 67):

> ```python
> ssh_key=os.path.expanduser(data["ssh_key"]),
> ```

Under Windows Python, `~` expands to the Windows home directory (e.g., `C:\Users\...`), not the WSL home. This breaks SSH key resolution. There is no detection or warning for this case.

### 8. No SSH agent forwarding support (Minor)

The SSH configuration uses explicit key files via `-i`. There is no support for SSH agent forwarding (`-A`), which would be needed if the remote machine needs to access private git repositories or other SSH-authenticated services (e.g., for `git clone` during provisioning).

The `cmd_provision()` function does `git clone` on the remote side (line 274):

> ```python
> _ssh(config, f"git clone {remote_url} {config.remote_dir}", timeout=300)
> ```

If the remote URL is SSH-based (e.g., `git@github.com:...`), this will fail without agent forwarding or a deploy key on the remote machine.

---

## Summary Table

| # | Issue | Severity |
|---|-------|----------|
| 1 | No SSH retry logic | Critical |
| 2 | No SSH keepalive configuration | Critical |
| 3 | StrictHostKeyChecking=accept-new auto-trusts | Major |
| 4 | rsync has no retry, no --partial, no resume | Major |
| 5 | SSH key permissions not validated | Major |
| 6 | No SSH connection pooling or multiplexing | Minor |
| 7 | Windows path expansion breaks SSH key | Minor |
| 8 | No SSH agent forwarding support | Minor |

# Remote Infrastructure Audit: Consolidated Rollup

**Date:** 2026-02-25
**Author:** Root orchestrator session
**Sub-reports:**
- `docs/audit_remote_tmux_sessions.md` -- tmux lifecycle and session persistence
- `docs/audit_remote_ssh_resilience.md` -- SSH connection resilience
- `docs/audit_remote_process_supervision.md` -- process supervision and crash recovery
- `docs/audit_remote_bootstrap.md` -- remote environment setup
- `docs/audit_remote_run_postmortem.md` -- 2xH100 session reconstruction

---

## Executive Summary

The remote deployment infrastructure is a first-draft system that was sufficient to complete one training session through brute force (4 server restarts, manual intervention throughout, 64% GPU-time wasted on defects). It is not robust enough for unattended operation, automated recovery, or efficient multi-session use. The fundamental architectural gap is the complete absence of anything that monitors or restarts processes after failure. The system assumes a human operator is watching the SSH terminal at all times, ready to diagnose and fix issues manually. Every component -- tmux sessions, SSH connections, process lifecycles, bootstrap -- lacks the retry/recovery semantics needed for spot instance workflows where interruption is not exceptional but expected.

**Total issues found across all 5 audits: 39**

---

## Issue Inventory by Severity

### Critical (10 issues)

These issues cause data loss, silent failures, or make the system fundamentally unsuitable for its stated purpose (spot instance training).

| # | Issue | Source Report | Impact |
|---|-------|-------------|--------|
| C1 | No process supervisor (systemd, supervisord, etc.) | Process Supervision | Crashed processes stay dead. No automatic recovery. |
| C2 | No health checking or heartbeat monitoring | Process Supervision | Server death goes undetected until training script hits it. |
| C3 | VM crash/preemption has no automatic recovery | Process Supervision | Entire session lost. Manual re-provision and re-launch required. |
| C4 | No SSH retry logic | SSH Resilience | Any transient network issue crashes the operation immediately. |
| C5 | No SSH keepalive configuration | SSH Resilience | Long SSH operations silently die from NAT/firewall timeouts. |
| C6 | No session reconnection logic in remote.py | tmux Sessions | After network drop, no workflow to reattach to running work. |
| C7 | Unconditional kill-before-create destroys running work | tmux Sessions | Re-running launch commands kills in-progress training. |
| C8 | No system-level dependency verification | Bootstrap | tmux, rsync, git not checked. First SSH command fails if missing. |
| C9 | ZMQ still listed as critical import (dead architecture) | Bootstrap | Bootstrap fails if zmq can't compile, but zmq isn't needed. |
| C10 | launch_report.json is local, not from H100 run | Postmortem | Remote validation artifacts not preserved. No evidence of remote validation. |

### Major (14 issues)

These issues cause significant waste, require manual workarounds, or indicate design problems that compound over time.

| # | Issue | Source Report | Impact |
|---|-------|-------------|--------|
| M1 | Two competing tmux management systems | tmux Sessions | Confusion, different behaviors, neither fully integrated. |
| M2 | No environment setup within tmux sessions | tmux Sessions | Wrong Python/libs possible on cloud VMs. |
| M3 | pipe-pane logging set up after session creation | tmux Sessions | Early output (including crash messages) may be lost. |
| M4 | No cleanup verification on teardown failure | tmux Sessions | Orphan sessions survive failed teardown. |
| M5 | StrictHostKeyChecking=accept-new auto-trusts | SSH Resilience | First connection to any host auto-trusted. MITM risk. |
| M6 | rsync has no retry, no --partial, no resume | SSH Resilience | Interrupted 5.8GB model transfer restarts from zero. |
| M7 | SSH key permissions not validated | SSH Resilience | SSH silently refuses keys with 0644 permissions. |
| M8 | Training checkpoint resume is manual-only | Process Supervision | Crashed training requires human to `--resume`. |
| M9 | Signal handling local only, not remote | Process Supervision | SIGHUP from tmux kill may not flush final checkpoint. |
| M10 | distributed.py is completely unused | Process Supervision | Dead code. Multi-GPU uses per-server architecture, not NCCL. |
| M11 | Bootstrap not fully idempotent (re-quantizes) | Bootstrap | Re-running bootstrap wastes GPU-minutes re-quantizing. |
| M12 | Hardcoded paths and remote filesystem assumptions | Bootstrap | Path resolution breaks on non-standard layouts. |
| M13 | uv installation via pip is fragile | Bootstrap | May fail on minimal cloud VMs with old pip/Python. |
| M14 | 8 deferred defects from 2026-02-16 still unresolved | Postmortem | No `remote.py generate/train`, no server readiness, no auto-discovery. |

### Minor (8 issues)

These are usability problems, code quality issues, or risks that haven't manifested yet.

| # | Issue | Source Report |
|---|-------|-------------|
| m1 | Log file accumulation without rotation | tmux Sessions |
| m2 | Session names not globally unique | tmux Sessions |
| m3 | No tmux scrollback buffer configuration | tmux Sessions |
| m4 | No SSH connection pooling or multiplexing | SSH Resilience |
| m5 | Windows path expansion breaks SSH key | SSH Resilience |
| m6 | No SSH agent forwarding support | SSH Resilience |
| m7 | No PID files or lock files | Process Supervision |
| m8 | No GPU memory verification after crash | Process Supervision |

### Additional Minor (from Bootstrap and Postmortem)

| # | Issue | Source Report |
|---|-------|-------------|
| m9 | No CUDA driver version verification | Bootstrap |
| m10 | Model download lacks integrity verification | Bootstrap |
| m11 | rsync exclude list not an include whitelist | Bootstrap |

---

## Cross-Cutting Patterns

### Pattern 1: "No code anywhere handles reconnection"

This is the single most pervasive gap. It manifests in every subsystem:

- **SSH**: No retry. One failure = crash.
- **tmux**: No reconnect workflow. Two competing scripts, neither handles "session exists, reattach."
- **Process supervision**: No restart. Crash = dead forever.
- **Training**: No automatic resume. Crash = manual `--resume` invocation.
- **rsync**: No `--partial`. Interrupted transfer = restart from zero.

The infrastructure was designed for the happy path: everything starts, everything stays running, the operator watches continuously, and the session completes without interruption. Every departure from this ideal requires manual intervention.

### Pattern 2: "Destructive defaults"

Multiple components default to destroying existing state rather than preserving it:

- `_remote_tmux_launch()` kills existing sessions unconditionally before creating new ones
- `remote_node_bootstrap.py` re-quantizes existing FP8 models
- rsync without `--partial` deletes partially transferred files
- No lock files prevent concurrent operations on the same resources

### Pattern 3: "Local works, remote breaks"

The first remote run (case study) surfaced 4 cross-platform portability defects and 3 SSH/environment defects that never appeared during local development. The infrastructure was tested only on the developer's Windows/WSL2 machine. There is no integration test that exercises the remote workflow even against a local SSH target.

### Pattern 4: "Dead architecture leaking into live code"

ZMQ is declared dead in CLAUDE.md, but:
- `zmq` is in the critical imports list for bootstrap validation
- `remote_node_bootstrap.py` uses ZMQ for server health checking
- `launch_remote.py` launches `futudiffu.server` (ZMQ-based, not FastAPI-based)
- `cmd_launch` in `remote.py` launches `futudiffu.server` (ZMQ-based)
- The entire `scripts/train.py` depends on ZMQ client/server

The src_ii/ rewrite includes FastAPI server and HTTP client, but the remote infrastructure still targets the old ZMQ architecture. A remote run using the new architecture is not currently possible without manual intervention.

### Pattern 5: "Documentation exceeds implementation"

The documentation is thorough and accurate. `remote_deployment.md` correctly describes what survives preemption and what doesn't. `case_study_2xh100_20260216.md` is an exemplary postmortem. `remote_session_agenda.md` has a clear operational plan. `remote_training_plan.md` has detailed cost estimates.

But the implementation lags far behind the documentation. The docs describe workflows that require manual SSH sessions. The automation tooling (`remote.py`) covers provisioning and launch but not the actual training workflow. The gap between "documented plan" and "automated capability" is where the 64% GPU-time waste occurred.

---

## Prioritized Fix List

### Tier 1: Must-fix before next remote run

These prevent the most expensive failure modes and could be implemented in a single focused session.

1. **SSH retry with exponential backoff in `_ssh()`** -- 3 retries, 2/4/8 second delays. Eliminates transient failure crashes. (~20 lines changed)

2. **SSH keepalive in `_ssh_args()`** -- Add `ServerAliveInterval=30` and `ServerAliveCountMax=3`. Prevents NAT timeout disconnects. (~2 lines)

3. **SSH ControlMaster multiplexing** -- Add `ControlMaster=auto`, `ControlPath=/tmp/fd_ssh_%h`, `ControlPersist=600`. Eliminates rate limiting by reusing a single TCP connection. (~3 lines)

4. **rsync --partial --partial-dir=.rsync-partial** -- Interrupted transfers resume instead of restarting. (~1 line per rsync call)

5. **Replace kill-before-create with check-before-create in `_remote_tmux_launch()`** -- Check `tmux has-session` first. If session exists, print warning and either reattach or require explicit `--force` to kill. (~10 lines)

6. **Remove `zmq` from critical imports** -- Replace with `fastapi` or `uvicorn` in bootstrap validation. (~2 lines)

7. **Auto-chmod 600 SSH key** -- In `RemoteConfig.load()`, check permissions and fix them. (~5 lines)

### Tier 2: Should-fix for operational efficiency

These reduce manual intervention and improve the development cycle.

8. **`remote.py status --health`** -- Add a health check that SSHes in, checks each tmux session exists, checks each server port is open, and reports a summary. Single command for "is everything still alive?"

9. **Unify tmux management** -- Merge `remote_tmux.py` capabilities into `remote.py`. One tool, one set of behaviors. `remote_tmux.py` has the better design (has-session checking, refuses to clobber), `remote.py` has the better integration (SSH transport, config-driven).

10. **`remote.py generate` and `remote.py train`** -- Wrap the training workflow in the same pattern as `cmd_validate` and `cmd_launch`. Reads topology from `remote_target.json`, constructs the correct command line, launches in tmux.

11. **Server readiness sentinel** -- Server writes a file (e.g., `.ready_PORT`) on startup. Wait logic polls for file existence instead of TCP connect, which is cheaper and more reliable.

12. **Bootstrap system dependency check** -- Before uv sync, verify: `which tmux`, `which rsync`, `which git`, `nvidia-smi` runs. Fail early with actionable error messages.

### Tier 3: Nice-to-have for production use

These are for sustained multi-session, multi-run operation.

13. **Simple watchdog script** -- A tmux session that periodically checks if other expected sessions are alive and restarts them if dead. Not systemd, just a bash loop with `tmux has-session` checks.

14. **Automatic checkpoint frequency for spot instances** -- Training should checkpoint every 5 minutes by wall clock, not every N steps. Use `time.monotonic()` to trigger checkpoints independently of training cadence.

15. **Run ID in session names** -- `fd_{run_id}_server_0` instead of `fd_server_0`. Prevents cross-run confusion.

16. **Log rotation or timestamped logs** -- `~/.futudiffu_logs/{session}_{timestamp}.log` instead of appending to a single file.

17. **Update remote infrastructure to target src_ii/ FastAPI server** -- `cmd_launch` should use `scripts_ii/launch_server.py`, not `futudiffu.server`.

---

## Effort Estimates

| Tier | Items | Estimated effort | Impact |
|---|---|---|---|
| Tier 1 (must-fix) | 7 items | 2-3 hours | Eliminates ~80% of infrastructure-caused waste |
| Tier 2 (should-fix) | 5 items | 4-6 hours | Reduces manual intervention from "constant" to "occasional" |
| Tier 3 (nice-to-have) | 5 items | 4-8 hours | Enables semi-autonomous multi-hour spot runs |

The first remote run burned ~83 minutes of GPU time on defects. At $1/GPU-hr for 2 GPUs, that is approximately $2.77. Tier 1 fixes would prevent the majority of this waste in future runs. The implementation cost (2-3 hours of developer time) pays for itself within 1-2 remote sessions.

---

## Conclusion

The remote infrastructure is a prototype that achieved its goal (completing one training run) but is not suitable for repeated use without significant manual babysitting. The good news: the documentation and case study provide an unusually clear picture of what went wrong and why. The codebase has all the building blocks (tmux management, rsync patterns, checkpoint resume) -- they just need to be wired together with retry semantics, health checking, and reconnection awareness. The Tier 1 fixes are small, mechanical changes that address the highest-impact failure modes.

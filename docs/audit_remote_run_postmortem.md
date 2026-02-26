# Audit: Remote Run Postmortem (2xH100, 2026-02-16)

**Date:** 2026-02-25
**Scope:** Reconstruction of what happened during the first remote training session.
**Files examined:** `docs/case_study_2xh100_20260216.md`, `docs/live_run_defects_20260216.md`, `docs/remote_session_agenda.md`, `docs/remote_deployment.md`, `remote_validation/launch_report.json`, `remote_validation/server_gpu0.log`

---

## Executive Summary

The first remote 2xH100 training session on PrimeIntellect spot required 4 server restart cycles over approximately 79 minutes of total wall clock before achieving a successful 28.7-minute training run. 23 defects were discovered, of which 15 were fixed live during the session and 8 were deferred. The total cost of defect-caused waste was approximately 83 minutes of GPU time, roughly equal to the duration of the successful training run itself. The most expensive defects were in torch.compile interaction (recompilation from LoRA injection: ~20 min wasted) and ZMQ concurrency bugs (~30 min wasted across two separate manifestations). The infrastructure itself (tmux sessions, rsync, SSH) contributed defects #8 (SSH key permissions), #14 (Windows path expansion), and #17 (no pipe-pane logging), totaling approximately 10 minutes of wasted time but representing deeper systemic issues with the remote workflow.

**The launch_report.json in the repository is from a LOCAL RTX 4090 run (2026-02-15), not from the remote H100 session. The remote session's validation artifacts were not preserved in the repository.**

---

## Timeline Reconstruction

### Run 1 (~15 min) -- Immediate platform failures

**What happened:**
1. SSH connection established to PrimeIntellect 2xH100 instance
2. `train.py` crashed immediately: hardcoded Windows path `r"F:\dox\repos\ai\futudiffu\src"` on line 35
3. Fixed: changed to `Path(__file__).resolve()`
4. `.supersekrit` file contained a comment line (`# what is this...`) that was read as the HF token, causing 401 errors on HF uploads
5. SSH key `~/.ssh/primeintellect_ed25519` had 0644 permissions, SSH refused it silently
6. Triton `continue` statement unsupported on SM90 in sage_kernels.py

**Root causes:** All portability failures. Code was tested only on the developer's Windows/WSL2 environment.

**Time cost:** ~15 min (catch, fix, rsync patch, restart)

### Run 2 (~25 min) -- torch.compile interaction + ZMQ race

**What happened:**
1. Server started successfully on both GPUs
2. `inject_lora` for ptheta triggered full torch.compile recompilation: 14+ minutes of idle H100s
3. Root cause: `inject_lora` conflates graph mutation and weight initialization (defect #18)
4. Concurrently: `LoRALinear.forward()` n==1 fast path exposed 192 matmul nodes to inductor, making compile time superlinear
5. ZMQ broadcast race: `_broadcast_return_primary()` only waited for client[0]'s future. Server 1 never received `allocate_adapter("ptheta")`. Diagnosed from VRAM discrepancy in `nvidia-smi` (10,735 MiB vs 9,633 MiB)

**Root causes:** API design (conflated operations), compiler interaction (visible matmuls), concurrency (ZMQ socket state machine)

**Time cost:** ~25 min

### Run 3 (~10 min) -- ZMQ EFSM concurrent access

**What happened:**
1. Previous ZMQ broadcast race was fixed
2. `generate_trajectories()` round-robin dispatched 4 jobs across 2 servers
3. Jobs 0 and 2 both submitted to ThreadPoolExecutor for the same server
4. Pool ran them concurrently on different threads, causing concurrent access to the same ZMQ REQ socket
5. ZMQ EFSM error: "Operation cannot be accomplished in current state"

**Root cause:** ThreadPoolExecutor has no thread affinity. Per-client locks were needed.

**Time cost:** ~10 min

### Run 4 (28.7 min) -- Success

**Timeline of successful run:**

| Phase | Wall clock | Notes |
|---|---|---|
| Prompt encoding | 71.6s | 24 prompts + negative, TE freed after |
| Adapter allocation | ~0.1s | rtheta (6 slots) + ptheta (102 slots) |
| Compile warmup | 42.5s | Both GPUs, all adapter slots pre-allocated |
| BTRM training | ~240s | 30 macrobatches |
| Policy optimization | ~1400s | 50 iterations |
| **Total** | **28.7 min** | |

**Key fix that enabled success:** Per-client `threading.Lock` instances on MultiGPUClient. All pool-dispatched methods acquire the target client's lock before touching its ZMQ socket. This serialized operations per-server while allowing cross-server parallelism.

---

## Defect Inventory

### By Category

| Category | Count | Fixed Live | Deferred |
|---|---|---|---|
| Cross-platform portability | 4 | 4 | 0 |
| Concurrency / thread safety | 2 | 2 | 0 |
| torch.compile interaction | 3 | 2 | 1 |
| API design (conflated operations) | 2 | 1 | 1 |
| Operational (missing automation) | 7 | 3 | 4 |
| Data persistence | 2 | 2 | 0 |
| Kernel portability (SM89->SM90) | 1 | 1 | 0 |
| Provision hygiene | 2 | 0 | 2 |

### The 8 Deferred Defects (Still Open)

1. **No `update_btrm_head` RPC** -- Cannot load trained BTRM head weights onto a different server via RPC. Blocks mid-session state replication across GPUs.

2. **No auto-discovery of server count** -- Operator must explicitly pass `--ports 5555 5556`. Easy to forget a GPU.

3. **No `remote.py generate` / `remote.py train` commands** -- Every generation/training launch was manual SSH + tmux. Flag names wrong, wrong flags used.

4. **No server readiness signal** -- Sleep-loop polling only. Server should write a sentinel file or accept a `--notify-fd`.

5. **No `--model-dir` auto-discovery** -- Typing wrong filenames wastes a full server restart cycle.

6. **No `remote.py upload` subcommand** -- Manual tmux launch for HF upload.

7. **`inject_lora` not formally deprecated** -- Convenience wrapper exists but the split into allocate/init is not enforced.

8. **Provision rsync whitelist** -- Carpet-bomb provisioning sends ~400MB of unnecessary data to the remote.

**Assessment:** Defects 2, 3, 4, 5, 6, and 8 are infrastructure/automation defects that directly relate to the remote deployment workflow. They were identified during the first run but remain unaddressed. The remote infrastructure has not been updated since the 2026-02-16 session.

---

## What Worked

1. **register_buffer for lora_scale** -- Phase transitions were graph-invariant. Saved ~28 min of recompilation.
2. **custom_op with register_autograd** -- All 12 ops had backward passes. Training worked on first forward.
3. **Dataset v2 sealed blobs** -- No corruption from partial writes.
4. **JSON envelope protocol** -- Cross-architecture portability (SM89 client -> SM90 server).
5. **JSONL metrics** -- Complete audit trail, structured queries for diagnosis.
6. **Health check renders** -- All 6 policy renders passed.
7. **model_manager.py VRAM lifecycle** -- Clean server restarts, no leaked GPU memory.
8. **Live defects document** -- Accumulated context across 4 restarts, preventing re-diagnosis of known issues.

---

## Documented vs Undocumented Issues

### Documented (in case study + live defects)
All 23 defects are documented with root cause analysis and fix descriptions.

### Undocumented but observed
1. **PrimeIntellect SSH rate limiting** -- Mentioned in defect #17 as the reason pipe-pane logging was needed ("SSH rate limiting causes Connection reset by peer"), but no mitigation was implemented or documented.
2. **Session management was entirely manual** -- The case study describes 4 server restarts but never mentions using `remote.py` or `remote_tmux.py` for session management. The restarts were likely done via manual SSH + tmux commands, suggesting the orchestration tooling was not actually used during the run.
3. **launch_report.json is local, not remote** -- The `remote_validation/launch_report.json` in the repository records a local RTX 4090 validation run from 2026-02-15 (timestamp: `2026-02-15T16:52:28`, GPU: `NVIDIA GeForce RTX 4090`, model paths: Windows `F:\dox\...`). The remote H100 session's validation artifacts were not pulled back or were overwritten.

---

## Cost Analysis

| Activity | GPU-minutes | Cost (est. $1/GPU-hr) |
|---|---|---|
| Run 1 (platform failures) | 2 GPUs * 15 min = 30 | $1.00 |
| Run 2 (compile + ZMQ) | 2 GPUs * 25 min = 50 | $1.67 |
| Run 3 (EFSM) | 2 GPUs * 10 min = 20 | $0.67 |
| Run 4 (success) | 2 GPUs * 28.7 min = 57.4 | $1.91 |
| **Total** | **157.4 GPU-min** | **$5.25** |

Approximately 64% of GPU time was wasted on defects. The successful run consumed 36% of total GPU-minutes.

---

## What Would Need to Change for the Next Remote Run

### Infrastructure (from this audit series)
1. SSH retry logic and keepalive
2. Process supervision (at minimum: tmux session restart on crash)
3. tmux session reconnection awareness
4. System dependency verification in bootstrap
5. SSH connection pooling to avoid rate limiting
6. rsync --partial for model transfers

### Application (from deferred defects)
1. `remote.py generate` and `remote.py train` commands
2. Server readiness sentinel file
3. Model path auto-discovery
4. BTRM head replication RPC
5. Provision rsync whitelist

### Architecture (from case study lessons)
1. Migrate from ZMQ to FastAPI/HTTP (partially done in src_ii/)
2. FlexAttention shape normalization for zero recompile spikes
3. Automated checkpoint frequency for spot instances

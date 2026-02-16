# Live Run Defects — 2xH100 Session 2026-02-16

Defects discovered during the first remote 2xH100 training session.
Each is actionable between runs.

## Critical: Multi-GPU training broken by design

### 1. MultiGPUClient broadcasts generation but not mutation RPCs
- `inject_lora`, `inject_btrm_head`, `set_adapter_config` route to primary only
- `generate_trajectories` distributes `sample_trajectory` across all servers
- Result: worker servers generate rollouts without LoRA adapters or BTRM heads
- **Fix**: Broadcast inject/config RPCs to all servers. Training-step RPCs (gradient accumulation, optimizer step) stay primary-only. Add `sync_btrm_to_all()` alongside existing `sync_lora_to_all()`.

### 2. No `update_btrm_head` RPC
- Can dump BTRM head weights to disk via `dump_all_loras`
- Cannot load trained BTRM head weights onto a different server via RPC
- Blocks mid-session state replication across GPUs
- **Fix**: Add `update_btrm_head(state_dict)` to client, server, and model_manager.

### 3. `inject_lora` crashes on duplicate adapter name
- `LoRALinear.add_adapter()` raises `ValueError` if name exists
- Blocks `--resume` after crash: Phase 2 tries to inject ptheta which already exists on primary
- **Fix**: Add `if name in self.adapters: return self.adapters[name]` (idempotent injection).

## Operational: Default args cause silent single-GPU usage

### 4. `train.py --port` defaults to single GPU
- `--port 5555` creates `InferenceClient` (single server)
- Must explicitly pass `--ports 5555 5556` for `MultiGPUClient`
- Easy to "forget" a GPU. No warning when n_gpus > 1 but only 1 port given.
- **Fix**: Auto-discover server count from config or probe ports. Warn if known-multi-GPU but single port.

### 5. `generate_btrm_dataset.py` requires manual per-GPU launch
- `--server` accepts one endpoint
- Operator must manually spawn N processes with `--gpu-id 0`, `--gpu-id 1`
- **Fix**: Accept `--servers` (nargs="+"), auto-fork N workers with per-GPU staging dirs.

### 6. `--dataset-format` defaulted to v1 (legacy)
- v2 was written specifically to fix v1's problems
- Default should be v2; v1 is for backward compat only
- Fixed mid-session but caused wasted generation time.

## Orchestration: Manual steps that should be automated

### 7. No `remote.py generate` or `remote.py train` commands
- Every generation/training launch was ad-hoc SSH + tmux
- Flag names wrong (`--port` vs `--server`), wrong flags used
- **Fix**: `remote.py generate --t2i 100` and `remote.py train` that read topology from `remote_target.json`.

### 8. SSH key permissions not auto-fixed
- `~/.ssh/primeintellect_ed25519` had 0644, SSH refused it
- **Fix**: Auto-chmod 600 in remote.py before first SSH use.

### 9. `.supersekrit` comment line not handled
- File contains `# what is this...` comment
- `read_text().strip()` grabbed the comment as the token
- Fixed mid-session in upload_to_hf.py and launch_remote.py.

### 10. PNGs in UPLOAD_EXTENSIONS
- Render PNGs were uploaded to HuggingFace dataset
- Renders should be local-only or regenerated from trajectory data
- Fixed mid-session by removing `.png` from UPLOAD_EXTENSIONS.

### 11. Triton `continue` unsupported on SM90
- sage_kernels.py used `continue` in Triton jit loops
- Triton 3.6.0 doesn't support it
- Fixed mid-session with `do_block` flag pattern.

### 12. Server startup blocks the operator
- Servers take ~30s to load models. No readiness signal.
- The operator (human or script) sits in a poll loop doing `sleep N && check`.
- **Fix**: Server should write a readiness sentinel file (e.g. `_ready_{port}`) or accept a `--notify-fd` that gets written to when ZMQ socket is bound. `remote.py launch --wait` should block on this sentinel, not sleep loops. Client `status()` call already works as a probe but requires the operator to know when to start probing.

### 13. Model filenames not in config
- Server launch command requires 3 model paths (`--fp8-diff`, `--te`, `--vae`)
- Filenames differ across environments (`zimage_nextdit_fp8` vs `z_image_fp8_blockwise`)
- Typing wrong filenames wastes a full server restart cycle
- **Fix**: Store model filenames in `remote_target.json` or discover from `models/` dir by extension/size. Server should have a `--model-dir` that auto-discovers.

### 14. Windows Python path expansion breaks SSH
- `remote.py` run via `.venv/Scripts/python.exe` expanded `~` to Windows home
- Must run via `/usr/bin/python3` for SSH key resolution
- **Fix**: remote.py should detect and refuse to run under Windows Python.

## Design: `inject_lora` conflates two distinct operations

### 18. `inject_lora` has dual behavior: graph mutation + weight initialization
- `inject_lora` does TWO things in one RPC:
  1. **Allocate memory, mutate compute graph** (add LoRALinear wrappers, register buffers,
     change module structure — invalidates torch.compile)
  2. **Initialize linear projections** (A/B weight init, set scale/alpha)
- These are fundamentally different operations with different costs and safety profiles:
  - (1) is expensive, irreversible mid-compile, must happen before warmup
  - (2) is cheap, can happen anytime, doesn't affect compiled graph
- Conflating them caused the 15+ minute recompilation waste: Phase 2 needed to call
  `inject_lora` for weight init, but that also triggered graph mutation + recompile.
- The pre-inject-all-at-startup workaround (defect #15 fix) masks the symptom but
  doesn't fix the root cause: any future adapter injection still forces recompile.
- **Fix**: Split into two RPCs:
  - `allocate_adapter(name, rank, layer_indices)` — graph-mutating, must happen before compile.
    Idempotent. Does not initialize weights.
  - `init_adapter_weights(name, init_b_std, alpha, scale)` — weight-only, graph-invariant.
    Can be called anytime without recompilation.
  - `inject_lora` becomes a convenience wrapper that calls both, with a deprecation warning
    if called after compile warmup.

### 19. `--skip-btrm` doesn't inject BTRM head for Phase 2 scoring
- `inject_btrm_head` only called inside `phase_btrm()`, which is skipped.
- Phase 2 rollouts use inline BTRM scoring (in `run_trajectory`) → server asserts `btrm_head is not None` → crash.
- Fallback `score_btrm` RPC also asserts `btrm_head is not None`.
- **Fix**: Inject BTRM head in main() when `--skip-btrm` and `run_policy`.
  If checkpoint exists, load trained BTRM weights; otherwise inject untrained head.
- Related: no `--btrm-checkpoint` flag to specify a checkpoint dir for `--resume` mode.
  Full `--resume` needs: load rtheta weights, load BTRM head weights, inject ptheta fresh.
- Fixed mid-session: inject untrained BTRM head when skipping BTRM (defect-aware testing).

### 20. MultiGPUClient broadcast race: ZMQ socket corruption
- Broadcast methods (`allocate_adapter`, `init_adapter_weights`, `inject_lora`, `inject_btrm_head`)
  submit tasks for ALL clients to `ThreadPoolExecutor`, but only wait for PRIMARY's result.
- If client[1]'s task from broadcast N is still in-flight when broadcast N+1 submits client[1]'s
  task, the pool may assign client[1]'s new task to a DIFFERENT worker thread.
- This causes concurrent access to the same ZMQ REQ socket → socket corruption → lost RPCs.
- **Observed**: Server 1 never received `allocate_adapter("ptheta")`. Server 1 compiled with
  only rtheta (6 adapters), while server 0 compiled with both (108 adapters). Server 1's
  warmup stuck because its ZMQ socket was corrupted.
- **Fix**: All broadcast methods must wait for ALL futures before returning.
  Extracted `_broadcast_return_primary()` helper that collects all results.
- **Root cause**: `ThreadPoolExecutor` has no thread affinity — any worker can pick up any task.
  Without waiting for all futures, there's no guarantee that a client's socket is idle when
  the next broadcast starts.
- **Extended fix**: Per-client `threading.Lock` instances on `MultiGPUClient`. ALL pool-dispatched
  methods (`generate_trajectories`, `_broadcast_return_primary`, `warmup_all`, `status_all`,
  `set_adapter_config`, `free`, `sync_lora_to_all`) acquire the target client's lock before
  touching its ZMQ socket. Jobs for the same server serialize; jobs for different servers
  run in parallel. Pool `max_workers` increased to `max(4, n_clients * 2)` to avoid deadlock
  from lock contention with limited threads.
- **Observed (run 3)**: `generate_trajectories` with group_size=4 across 2 servers dispatched
  job 2 (server 0) while job 0 (server 0) was still in-flight → ZMQ EFSM error
  "Operation cannot be accomplished in current state". The `_broadcast_return_primary` fix
  only covered broadcast RPCs; `generate_trajectories` round-robin was still unprotected.
- Fixed with per-client locks in run 4. Both GPUs now at 100% utilization.

### 21. LoRALinear n==1 fast path expanded 192 visible matmuls into compiled graph
- 96 modules with single adapter (ptheta-only) took the `if n == 1` fast path in
  `LoRALinear.forward()`, which used 2 explicit matmuls (visible to inductor).
- 192 visible matmul graph nodes triggered full GEMM analysis (tiling, memory layout,
  fusion decisions) in inductor. Compile time was superlinear in matmul count.
- Meanwhile, `multi_lora_op` (Triton custom_op with `register_autograd`) was already used
  for N>1 but bypassed for N=1 "for kernel launch overhead savings."
- **Fix (applied)**: Removed the n==1 fast path. ALL adapter counts (N>=1) now dispatch to
  `multi_lora_op`, which is opaque to inductor (zero GEMM analysis). Replaces 192 visible
  matmuls with 96 opaque custom_op calls.
- **Result**: Compile time dropped from 14+ min (stuck) → 55s → 42.5s across runs.
  First rollout: 47.5s (includes Triton kernel compile), subsequent: 3.9s.
- **Future**: Full packed-buffer refactor (pre-stacked A/B as nn.Parameters, eliminate
  per-forward `torch.stack`) would further reduce graph nodes but requires gradient masking
  for frozen adapters. Not needed at current scale.

## Performance: torch.compile recompilation on LoRA injection

### 15. Injecting LoRA after torch.compile forces full recompilation
- Phase 1 compiles model with rtheta (6 adapters). Phase 2 injects ptheta (102 adapters).
- New adapters invalidate compiled graph → torch.compile spawns 32 compile workers.
- 10+ minute recompilation with 0% GPU utilization. Both H100s sitting idle.
- Server 1's ptheta `inject_lora` also stalled (possibly blocks on dynamo guard checks).
- **Fix**: Pre-inject ALL adapters (rtheta + ptheta) with `scale=0.0` BEFORE first compile warmup.
  Toggle adapters via `set_adapter_config` (scale changes don't invalidate graphs since
  `lora_scale` is a registered buffer, not a Python attribute). This eliminates all
  mid-session recompilation.

### 16. `train.py` warmup calls `warmup()` not `warmup_all()`
- MultiGPUClient.warmup() only warms primary server.
- Server 1 takes first-compile penalty (53.95s) on first Phase 1 rollout.
- After Phase 2 injection, server 1 will take the full recompilation hit on its first rollout too.
- **Fix**: Call `warmup_all()` instead of `warmup()` in train.py Phase 2 transition.

### 17. No tmux pipe-pane or logfile tee for training/server output
- Training output only visible via `tmux capture-pane` requiring SSH round-trips.
- SSH rate limiting on PrimeIntellect causes `Connection reset by peer`.
- Operator must manually SSH to check training progress.
- Fixed mid-session with `tmux pipe-pane` → logfiles pulled by rsync loop.
- **Fix**: `remote.py launch` should set up pipe-pane to logfiles automatically.
  Training scripts should also tee stdout to a logfile directly (`tee -a train.log`).

## Data: Policy rollouts discarded after scoring

### 22. `phase_policy()` discards on-policy rollout latents after gradient computation
- `generate_trajectories()` returns full trajectory dicts (sparse steps + final).
- After BTRM scoring and gradient accumulation, `del trajectory` frees all tensors.
- On-policy rollouts are the highest-value training signal for future BTRM iterations:
  they cover the policy's actual distribution, not the off-policy generation distribution.
- No other code path captures these; the data is generated and destroyed in ~8 seconds.
- **Fix (applied)**: Added configurable rollout persistence to `phase_policy()`.
  `--persist-fraction` (default 0.01) and `--persist-min-k` (default 1) control how
  many rollouts are saved per iteration. Selects top-K by reward. Uses DatasetV2Writer
  (parquet index + safetensors blobs). Saved to `{output_dir}/policy_rollouts/`.
  Metadata includes: prompt, seed, cfg, iteration, reward, advantage, group rank.
  Default behavior: 1 rollout per iteration (the best) → 50 trajectories across
  50 iterations. ~3MB/traj × 50 = ~150MB. Negligible relative to training cost.

### 23. No HF upload tee for training output
- `training_output/` only visible via rsync pull loop (local-only).
- If the spot instance is preempted, any un-pulled data is lost.
- HuggingFace is the durable off-machine store for this project.
- **Fix (applied mid-session)**: Launched `upload_to_hf.py --watch` in tmux session
  `hf_upload` on the remote, watching `training_output/` with 60s poll interval.
  Repo: `SQCU/futudiffu-run01`. Confirmed uploading: metrics.jsonl, checkpoints,
  adapter safetensors. Future: `remote.py upload` subcommand to automate this.

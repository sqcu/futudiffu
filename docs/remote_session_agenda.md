# Remote Session Agenda: 2xH100 Validation + Training

## Goal

Validate 2xH100 pipeline end-to-end, confirm kernel correctness on SM90,
measure throughput, stream data to HF, then run real BTRM + policy training.

## Prerequisites

- `remote_target.json` configured with host, SSH key, remote_dir
- `.supersekrit` present in repo root (HF auth token)
- Local dataset at `btrm_dataset/` (50 trajectories, rsynced during provision)
- Inference server code passing all tests locally on SM89

---

## Phase 1: Provision + Validate (~15 min)

```bash
python scripts/remote.py provision
python scripts/remote.py bootstrap
python scripts/remote.py models --hf --wait
python scripts/remote.py validate --wait
```

**Gate**: launch_remote.py Phase 4 cosine similarity >= 0.90 (cross-arch SM89->SM90).

**Expected issues**:
- Kernel recompilation on first run (~45-73s per unique compile). Triton SM90
  codegen may surface new bugs not seen on SM89.
- ZMQ timeout during first compile warmup (server appears unresponsive while
  Triton compiles). Client auto-recovers via `_reset_socket()`.

**If kernel errors**: Edit locally, `remote.py patch`, restart affected tmux
session, repeat validation.

---

## Phase 2: Throughput + Resumability Stress Test (~15 min)

```bash
# Terminal 1: Start HF upload tee for generated data
python scripts/remote.py upload --repo-id SQCU/futudiffu-run01 --source-dir btrm_dataset

# Terminal 2: Generate trajectories
python scripts/remote.py ssh  # or use tmux directly
cd /path/to/repo && python scripts/generate_btrm_dataset.py --t2i 100 --output-dir btrm_dataset
```

**Purpose**: NOT dataset prep. This validates:

1. Wallclock per trajectory matches H100 expectations (~7.5s vs 22s on 4090)
2. HF upload tee is streaming (`upload_to_hf.py --watch` in parallel tmux session)
3. Dataset v2 writer handles abrupt `kill -9` without corrupting sealed blobs
4. Resume after kill picks up exactly where it left off (no duplicate traj IDs)
5. Multi-GPU parallelism if using both GPUs for generation

**Test procedure**:
- Start generation, let ~30 trajectories complete
- `kill -9` the generation process
- Verify HF has partial upload (check repo via `huggingface-cli`)
- Restart generation with same command
- Verify resume: next traj ID continues from where it stopped, no duplicates

---

## Phase 3: BTRM Training (~5 min)

```bash
python scripts/remote.py upload --repo-id SQCU/futudiffu-run01 --source-dir training_output

# In remote tmux:
python scripts/train.py \
    --ports 5555 5556 \
    --dataset-dir btrm_dataset \
    --output-dir training_output \
    --btrm-macrobatches 30 \
    --btrm-batch-size 32 \
    --render-every 8
```

Uses existing 50 trajectories (rsynced from local) + the 100 from Phase 2.
HF upload tee streams checkpoints as they're saved.

**Gate**: Loss decreasing, scrimble accuracy >70%, scrongle accuracy >70%.

---

## Phase 4: Policy Optimization (~20 min)

```bash
python scripts/train.py \
    --ports 5555 5556 \
    --dataset-dir btrm_dataset \
    --output-dir training_output \
    --skip-btrm --resume \
    --policy-iterations 50 \
    --policy-group-size 4 \
    --render-every 10
```

Policy rollouts use disjoint config space from BTRM training:
- Seeds: 70000+ (vs BTRM 1000-1099)
- Steps: 10 (vs BTRM [8-30])
- Attention: sage only (vs BTRM sdpa+sage)
- Prompts: all 24 (vs BTRM 4/seed)

This cross-validates BTRM generalization: policy asks "does the BTRM reward
model give useful signal on configs it hasn't trained on?"

**Gate**: Nonzero grad_norms, reward trend, render health checks pass.

---

## Phase 5: Pull + Teardown (~5 min)

```bash
python scripts/remote.py pull
# Verify HF repo has all artifacts
python scripts/remote.py teardown --pull
```

---

## HF Upload Strategy

Two tmux sessions running `upload_to_hf.py --watch`:

1. **hf_upload** (or `hf_upload_data`): watching dataset output dir (trajectory generation)
2. **hf_upload_train**: watching `training_output/` (adapters, metrics, renders)

Launch via:
```bash
python scripts/remote.py upload --repo-id SQCU/futudiffu-run01 --source-dir btrm_dataset
python scripts/remote.py upload --repo-id SQCU/futudiffu-run01 --source-dir training_output
```

Note: second invocation reuses the same tmux session name `hf_upload`. To run
both simultaneously, manually launch the second via SSH with a different
session name, or extend `cmd_upload` with a `--session-name` arg.

---

## Expected Bugs

| Bug | Mitigation |
|---|---|
| Triton SM90 kernel compilation failures | Edit locally, `remote.py patch`, restart |
| ZMQ timeout on first compile warmup | Client `_reset_socket()` auto-recovers |
| Lifecycle transition VRAM waste | Accept for now; `--no-offload` not yet implemented |
| `.supersekrit` missing on remote | Now provisioned via rsync (removed from exclude list) |
| Hardcoded Windows paths in scripts | Fixed: all scripts use repo-relative `Path(__file__)` |

---

## Why This Document Exists

Context compaction during a multi-hour remote training session will lose the
rationale for why we're doing each step. This document is the durable
reference that survives context window compression.

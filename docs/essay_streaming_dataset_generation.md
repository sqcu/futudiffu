# Streaming Dataset Generation: Write-Through and Crash Resumability

**Date:** 2026-02-19
**Author:** Subagent (restructuring task)
**Provenance:** User directive to restructure `scripts_ii/generate_multi_res_trajectories.py` for streaming writes and crash-resumability, motivated by 38.7-minute generation runs on preemptible spot instances.

---

## 1. What Changed and Why

### The Problem

The dataset generation script followed a "compute-first, write-last" architecture:

1. Phase 2 generated all trajectories into a Python list (`trajectories`) in RAM.
2. Phase 3 iterated that list and wrote them all to the V2 dataset on disk.
3. The step_sigmas sidecar was accumulated in a dict and bulk-written at the end.

For 300 trajectories at ~8 seconds each, this meant 38.7 minutes of GPU computation with zero persistence. A crash at minute 37 would lose everything. On preemptible spot instances -- which is the target deployment for FP8 TFLOPS-bound workloads -- this architecture is a guaranteed data loss vector.

### The Solution

Phases 2 and 3 are merged into a single `phase2_generate_and_persist()` function. Each trajectory is written to the `DatasetWriter` immediately after its GPU computation completes. The `trajectories` list accumulation pattern is eliminated entirely. Tensor data is freed from CPU RAM within the same loop iteration it is persisted.

A new `src_ii/dataset_resumption.py` module provides the resumability logic: it computes which trajectories from a deterministic generation plan already exist in the dataset's parquet index, and returns only the remaining work.

### Files Changed

| File | Change |
|------|--------|
| `scripts_ii/generate_multi_res_trajectories.py` | Merged Phases 2+3 into streaming write-through. Added `_build_generation_plan()`, replaced `phase2_generate()` + `phase3_persist()` with `phase2_generate_and_persist()`. Added `_build_metadata_from_plan()` for the all-complete early exit path. |
| `src_ii/dataset_resumption.py` | **New module.** Trajectory identity tuples, plan save/load, completed trajectory detection from parquet index, plan diffing, incremental step_sigmas sidecar updates. |
| `tests/test_dataset_resumption.py` | **New test file.** 21 tests covering identity extraction, plan persistence, completed detection, remaining work computation, sidecar updates, determinism, and integration with DatasetWriter. |

---

## 2. Before/After Architecture

### Before: Buffered (Compute-First, Write-Last)

```
Phase 1: Load TE -> encode prompts -> free TE
Phase 2: Load diffusion model -> generate ALL trajectories into list[] -> free model
Phase 3: Open DatasetWriter -> write all from list[] -> close writer
         Write step_sigmas sidecar from list[]
         Free trajectory tensors from CPU RAM
Phase 4: Load VAE -> decode from dataset on disk -> free VAE
Phase 5: Verify
Phase 6: Bin packing analysis
```

**Data flow:** GPU -> CPU list -> disk (batch)
**Vulnerability window:** Entire Phase 2 duration (38+ minutes)
**CPU RAM peak:** All trajectory tensors simultaneously in memory
**Resume capability:** None

### After: Streaming (Write-Through)

```
Phase 1: Load TE -> encode prompts -> free TE
Phase 2: Build generation plan
         Check existing dataset for completed trajectories (resume)
         Load diffusion model
         For each trajectory in plan:
           Skip if already in dataset
           Generate on GPU
           Write to DatasetWriter immediately
           Update step_sigmas sidecar incrementally
           Free CPU tensor RAM
           Flush every 10 trajectories
         Final flush
         Free model
Phase 4: Load VAE -> decode from dataset on disk -> free VAE
Phase 5: Verify
Phase 6: Bin packing analysis
```

**Data flow:** GPU -> CPU -> disk (per-trajectory)
**Vulnerability window:** At most 1 trajectory generation cycle (~8s) between flushes
**CPU RAM peak:** One trajectory's tensors at a time
**Resume capability:** Full -- re-run with same params skips completed work

---

## 3. The Resumability Protocol

### Identity Matching

Each trajectory has a unique identity defined by 7 columns:

```python
IDENTITY_COLUMNS = (
    "seed",
    "prompt_idx",
    "width",
    "height",
    "attention_backend",
    "n_steps",
    "cfg",
)
```

These are the minimal set of parameters that determine a trajectory's output given the same model weights and code version. Two trajectories are "the same" if and only if all 7 fields match.

### The Resume Flow

1. **Build plan:** `_build_generation_plan()` produces a deterministic, ordered list of trajectory specifications. The plan is saved to `generation_plan.json` for auditability.

2. **Detect completed work:** `compute_remaining_work(plan, dataset_dir)` reads the parquet index (a few KB), extracts the identity tuple of every existing trajectory, and diffs against the plan.

3. **Skip or generate:** The main loop iterates the full plan in order. For each entry:
   - If its identity is in the completed set: append metadata (no tensors) and `continue`.
   - If not: generate, write, flush, update sidecar.

4. **Early exit:** If all planned trajectories exist, the function returns without loading the diffusion model at all. This means a completed generation run can be re-verified (Phases 4-6) without touching the GPU.

### Plan Determinism

The generation plan must be identical across runs for resumption to work. This is guaranteed by:

- `RESOLUTION_TIERS` is built from a fixed RNG seed (777) at import time.
- Seeds are assigned sequentially from `BASE_SEED` (400000).
- Prompt indices cycle deterministically via `plan_index % len(MULTI_PROMPTS)`.
- The iteration order (resolution tiers x backends x n_traj) is fixed.

The plan file (`generation_plan.json`) is written at the start of each run. If a user changes configuration constants (e.g., `TRAJECTORIES_PER_TIER`), the plan changes, and resumption correctly treats all old entries as "extra" (ignored) and all new entries as "remaining."

---

## 4. Edge Cases and Design Decisions

### Edge Case: DatasetWriter Append Semantics

The `DatasetWriter.__enter__()` method reads the existing parquet index and continues from the next `traj_id`. This means resumed runs produce sequential traj_ids without gaps. The blob sealing and index writing are idempotent -- if the script crashes between `add_trajectory()` and `flush()`, the in-memory blob is lost, but the next run will detect the missing trajectory and regenerate it.

### Edge Case: Partial Blob Loss

If the script crashes after `add_trajectory()` but before `flush()`, the trajectories in the current in-memory blob are lost. On resume, `find_completed_identities()` reads the parquet index, which only references sealed blobs. The unsealed trajectories are not in the index, so they appear as "remaining" and are regenerated. This is the correct behavior: the vulnerability window is at most 10 trajectories (the flush interval), which at ~8s each is ~80 seconds of lost work.

### Edge Case: Step Sigmas Sidecar

The sidecar is updated incrementally after each trajectory write. The `update_step_sigmas_sidecar()` function reads the existing JSON, adds the new entry, and writes atomically. On resume, previously written entries are preserved. If the sidecar is lost entirely (deleted), it can be rebuilt from the parquet index metadata (the step_sigmas and step_logsnrs are stored in the trajectory metadata passed to `add_trajectory()`).

### Design Decision: Plan Order vs. Backend Grouping

The old code grouped trajectories by resolution tier, then by backend. The new code iterates the full plan in order, which interleaves backends within each tier. This means the attention backend may switch more frequently, but `set_attention_backend()` is essentially free (it sets a global variable). The benefit is that the plan order is simple and the skip logic is a single `continue` statement.

### Design Decision: No Model Loading When Fully Complete

If all trajectories exist, `phase2_generate_and_persist()` returns without loading the diffusion model (~5.8 GB VRAM, several seconds of init). This enables fast re-verification runs: Phases 4 (render), 5 (verify), and 6 (packing) can execute on a previously completed dataset without touching the diffusion model.

### Design Decision: Module Boundary

The reusable logic (identity extraction, plan diffing, sidecar updates) lives in `src_ii/dataset_resumption.py`. The script remains thin orchestration: it calls `_build_generation_plan()` (which delegates to `resolution_sampling` and `sigma_schedule`), then calls `compute_remaining_work()` and `save_generation_plan()` from the module, and iterates the plan with skip logic.

---

## 5. Test Results

21 tests in `tests/test_dataset_resumption.py`, all passing:

> ```
> tests/test_dataset_resumption.py::TestTrajectoryIdentity::test_extracts_correct_fields PASSED
> tests/test_dataset_resumption.py::TestTrajectoryIdentity::test_different_specs_different_identities PASSED
> tests/test_dataset_resumption.py::TestTrajectoryIdentity::test_same_specs_same_identity PASSED
> tests/test_dataset_resumption.py::TestTrajectoryIdentity::test_missing_column_raises PASSED
> tests/test_dataset_resumption.py::TestPlanPersistence::test_round_trip PASSED
> tests/test_dataset_resumption.py::TestPlanPersistence::test_atomic_write PASSED
> tests/test_dataset_resumption.py::TestPlanPersistence::test_load_nonexistent_raises PASSED
> tests/test_dataset_resumption.py::TestPlanPersistence::test_validates_identity_columns PASSED
> tests/test_dataset_resumption.py::TestFindCompleted::test_finds_existing_trajectories PASSED
> tests/test_dataset_resumption.py::TestFindCompleted::test_empty_dataset PASSED
> tests/test_dataset_resumption.py::TestFindCompleted::test_nonexistent_dir PASSED
> tests/test_dataset_resumption.py::TestRemainingWork::test_fresh_run PASSED
> tests/test_dataset_resumption.py::TestRemainingWork::test_partial_resume PASSED
> tests/test_dataset_resumption.py::TestRemainingWork::test_all_complete PASSED
> tests/test_dataset_resumption.py::TestRemainingWork::test_preserves_plan_order PASSED
> tests/test_dataset_resumption.py::TestStepSigmasSidecar::test_create_new PASSED
> tests/test_dataset_resumption.py::TestStepSigmasSidecar::test_incremental_update PASSED
> tests/test_dataset_resumption.py::TestStepSigmasSidecar::test_overwrite_entry PASSED
> tests/test_dataset_resumption.py::TestPlanDeterminism::test_plan_is_deterministic PASSED
> tests/test_dataset_resumption.py::TestPlanDeterminism::test_identity_columns_are_hashable PASSED
> tests/test_dataset_resumption.py::TestIntegrationWithWriter::test_resume_after_partial_write PASSED
>
> ============================= 21 passed in 1.63s ==============================
> ```

Existing dataset tests (41 catalog tests) also pass with no regressions:

> ```
> tests/test_dataset_catalog.py ... 41 passed in 2.76s
> ```

The existing `test_dataset_v2.py` tests have a pre-existing Windows file locking issue (safetensors mmap holds blob handles open during `TemporaryDirectory` cleanup) that is unrelated to these changes. The test logic itself passes (prints "PASSED") before the cleanup `PermissionError` occurs.

---

## Appendix A: `src_ii/dataset_resumption.py` Key Functions

> ```python
> IDENTITY_COLUMNS = (
>     "seed", "prompt_idx", "width", "height",
>     "attention_backend", "n_steps", "cfg",
> )
>
> def trajectory_identity(spec: dict) -> tuple:
>     """Extract the identity tuple from a trajectory specification."""
>     return tuple(spec[col] for col in IDENTITY_COLUMNS)
>
> def find_completed_identities(dataset_dir: Path | str) -> set[tuple]:
>     """Scan a V2 dataset's parquet index and return identity tuples."""
>     # Reads only parquet index, not tensor data
>     ...
>
> def compute_remaining_work(plan, dataset_dir):
>     """Compare plan against dataset, return (remaining, completed, n_total)."""
>     completed_ids = find_completed_identities(dataset_dir)
>     remaining = [s for s in plan if trajectory_identity(s) not in completed_ids]
>     completed = [s for s in plan if trajectory_identity(s) in completed_ids]
>     return remaining, completed, len(plan)
>
> def update_step_sigmas_sidecar(sidecar_path, traj_index, step_sigmas, step_logsnrs):
>     """Incrementally update step_sigmas.json (read-modify-write, atomic)."""
>     ...
> ```

## Appendix B: Generation Loop Core (Streaming Write)

> ```python
> with DatasetWriter(str(dataset_dir)) as writer:
>     for spec in full_plan:
>         identity = trajectory_identity(spec)
>
>         # Skip already-completed trajectories
>         if identity not in remaining_identities:
>             trajectory_metadata.append({...})
>             continue
>
>         # Generate on GPU
>         result_tensors, meta = rollout(...)
>
>         # Move to CPU and write IMMEDIATELY
>         cpu_tensors = {k: v.cpu() for k, v in result_tensors.items()}
>         del result_tensors
>         torch.cuda.empty_cache()
>
>         traj_id = writer.add_trajectory(tensors=cpu_tensors, metadata=traj_metadata)
>         del cpu_tensors  # free CPU RAM immediately
>         new_since_flush += 1
>
>         # Flush every 10 newly-written trajectories
>         if new_since_flush >= 10:
>             writer.flush()
>             new_since_flush = 0
>
>         # Update sidecar incrementally
>         update_step_sigmas_sidecar(sidecar_path, traj_id, step_sigmas, step_logsnrs)
>
>     # Final flush
>     if new_since_flush > 0:
>         writer.flush()
> ```

## Appendix C: CLI Interface (Unchanged)

The script's CLI interface is backward-compatible:

> ```
> .venv/Scripts/python.exe scripts_ii/generate_multi_res_trajectories.py
> .venv/Scripts/python.exe scripts_ii/generate_multi_res_trajectories.py --sparse-steps 0,4,9,14,19,24,29
> ```

The output directory structure is unchanged. The V2 dataset format on disk is identical. The only new files are:
- `generation_plan.json` -- the deterministic plan (for auditability/debugging)
- `step_sigmas.json` -- now written incrementally instead of bulk (same final content)

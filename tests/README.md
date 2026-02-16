# tests/

Correctness and integration tests for the futudiffu inference and training
pipeline. Everything in this directory should be runnable as a standalone
script that either passes (exit 0) or fails (exit nonzero, ideally with a
clear error message).

## What belongs here

- Smoke tests that exercise a subsystem end-to-end and assert correctness.
- Integration tests that require the inference server to be running.
- Reproducibility tests that compare outputs against saved references.
- Unit-level correctness tests for kernels, LoRA mechanics, batch packing, etc.

## What does NOT belong here

- **Benchmarks.** Timing-only scripts go in `bench/`.
- **One-shot scripts** that generate data, render images, or push to HuggingFace.
  Those go in `scripts/`.
- **Library code.** If it defines reusable functions or classes, it belongs in
  `src/futudiffu/`.

---

*Intermezzo*

```
A codebase left ungardened grows to tangled vine,
Where tests and scripts commingle and no soul can draw the line.
The engineer who finds a smoke test hiding in the root
Will curse the name of whoever left it there without a suit.
So keep your tests in tests/, your benches on the bench --
Lest future-you must excavate this archaeological trench.
```

---

## Files

| File | Description |
|------|-------------|
| `smoke_test_btrm_v2.py` | BTRM + policy training with synthetic paired data (SDPA vs Sage trajectories) |
| `smoke_test_e2e_training.py` | End-to-end training smoke test via inference server RPCs |
| `test_e2e_real_trajectories.py` | BTRM + policy training against real stored trajectories from btrm_dataset/ |
| `test_flexattn_integration.py` | Full FlexAttention batch packing integration test on real FP8 model |
| `test_flexattn_packing.py` | Bit-for-bit verification of packed vs unpacked FlexAttention forward |
| `test_lifecycle_and_dump.py` | LoRA lifecycle persistence: inject, swap models, replay, crash dump |
| `test_lora_kernel_integration.py` | Multi-LoRA Triton kernel correctness vs cat-based reference path |
| `test_lora_save_load.py` | LoRA save/load/scale roundtrip across server restarts |
| `test_multilora_fused.py` | Fused scatter/gather multi-LoRA with concurrent score+sample batching |
| `test_repro_traj004.py` | Cross-session reproducibility test for trajectory 004 (bitwise comparison) |
| `test_scrongle_scrimble.py` | Two-head BTRM classification of step-count vs quantization artifacts |
| `run_compat_validation.py` | ComfyUI compatibility validation against reference tensor streams |

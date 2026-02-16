# scripts/

Standalone operational scripts for training, data generation, rendering,
analysis, and infrastructure. These are intended to be run manually or from
a tmux session -- they are not tests and they are not library code.

## What belongs here

- Training entrypoints (BTRM, policy optimization).
- Dataset generation and packing pipelines.
- Rendering and visualization scripts that produce images or plots.
- Analysis scripts that read saved tensors and compute statistics.
- Infrastructure utilities (remote bootstrapping, tmux management, uploads).
- Diagnostic one-offs that investigate specific pipeline behaviors.

## What does NOT belong here

- **Tests.** If it asserts correctness or exits nonzero on failure, put it in
  `tests/`.
- **Benchmarks.** If it only measures timing or throughput, put it in `bench/`.
- **Library code.** Reusable functions and classes belong in `src/futudiffu/`.

---

*Intermezzo*

```
Here rest the scripts that do the work of hands:
They train, they render, pack, and diagnose,
They bootstrap nodes across far-distant lands
And ship the fruits of labor where data flows.
Do not abandon scripts among the root --
A well-kept orchard bears the sweetest fruit.
```

---

## Files

| File | Description |
|------|-------------|
| `train.py` | Production training script: BTRM head training + policy optimization via server RPCs |
| `generate_btrm_dataset.py` | Schedule-driven diffusion trajectory generation for BTRM training data |
| `generate_policy_eval.py` | Generate evaluation trajectories with policy LoRA active for comparison |
| `pack_trajectories.py` | Pack per-trajectory directories into safetensors archives + JSONL manifest |
| `regen_stream_futudiffu.py` | Regenerate the canonical reference trajectory in stream_futudiffu/ |
| `render_trajectories.py` | VAE-decode stored trajectory latents to PNG via the inference server |
| `render_policy_comparison.py` | Generate side-by-side renders with policy LoRA on vs off |
| `remote_node_bootstrap.py` | Download models, quantize to FP8, and optionally validate on a remote node |
| `remote_tmux.py` | Tmux session management for persistent remote training runs |
| `validate_server_pipeline.py` | Validate FP8 inference server against reference trajectory |
| `upload_to_hf.py` | Upload dataset files to a HuggingFace repository (one-shot or watch mode) |
| `analyze_residuals.py` | Cross-session residual analysis for trajectory latents (pure CPU, no GPU) |
| `diagnose_i2i.py` | Measure per-step divergence of i2i trajectories from source images |
| `diff_renders.py` | False-color diff visualization across attention variant renders |

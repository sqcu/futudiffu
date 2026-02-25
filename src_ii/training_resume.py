"""Training resume detection and state loading.

Enables interruptible/resumable BTRM training. Checkpoints are written by
btrm_training.py at configurable intervals. This module detects the latest
checkpoint and loads its state.

Checkpoint directory layout (written by persist_btrm + training loop):
    output_dir/
        checkpoint_stepNNN/
            rtheta_adapter.safetensors
            btrm_head.safetensors
            btrm_compound_config.json
            optimizer_state.pt
            resume_state.json   # {step, rng_state, training_curve_len}
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch


def detect_resume(output_dir: str | Path) -> dict | None:
    """Scan output_dir for the latest checkpoint with a valid resume_state.json.

    Returns:
        Dict with keys {checkpoint_dir, step, rng_state, training_curve_len}
        or None if no valid checkpoint found.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return None

    candidates = []
    for d in sorted(output_dir.iterdir()):
        if not d.is_dir():
            continue
        m = re.match(r"checkpoint_step(\d+)", d.name)
        if m is None:
            continue
        resume_path = d / "resume_state.json"
        if not resume_path.exists():
            continue
        step = int(m.group(1))
        candidates.append((step, d, resume_path))

    if not candidates:
        return None

    # Pick latest by step number
    candidates.sort(key=lambda x: x[0], reverse=True)
    step, ckpt_dir, resume_path = candidates[0]

    with open(resume_path) as f:
        state = json.load(f)

    state["checkpoint_dir"] = str(ckpt_dir)
    state.setdefault("step", step)
    print(f"[training_resume] Found checkpoint at step {step}: {ckpt_dir}")
    return state


def load_optimizer_state(
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: str | Path,
) -> None:
    """Load optimizer state dict from a checkpoint directory."""
    opt_path = Path(checkpoint_dir) / "optimizer_state.pt"
    if not opt_path.exists():
        print(f"[training_resume] No optimizer state at {opt_path}, starting fresh")
        return

    state = torch.load(str(opt_path), map_location="cpu", weights_only=True)
    optimizer.load_state_dict(state)
    print(f"[training_resume] Loaded optimizer state from {opt_path}")


def save_resume_state(
    output_dir: str | Path,
    step: int,
    optimizer: torch.optim.Optimizer,
    training_curve_len: int = 0,
) -> Path:
    """Save optimizer state + resume metadata to a checkpoint directory.

    Call this AFTER persist_btrm() has already written adapter + head weights
    to the same checkpoint directory.

    Args:
        output_dir: Checkpoint directory (e.g. output/checkpoint_step050).
        step: Current training step.
        optimizer: Optimizer whose state to save.
        training_curve_len: Number of entries in the training curve JSONL.

    Returns:
        Path to the resume_state.json file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save optimizer state
    opt_path = output_dir / "optimizer_state.pt"
    torch.save(optimizer.state_dict(), str(opt_path))

    # Save resume metadata
    resume_state = {
        "step": step,
        "training_curve_len": training_curve_len,
    }
    resume_path = output_dir / "resume_state.json"
    with open(resume_path, "w") as f:
        json.dump(resume_state, f, indent=2)

    print(f"[training_resume] Saved resume state at step {step} to {output_dir}")
    return resume_path

"""Dataset loading and example construction from stored BTRM trajectories.

Loads trajectories from btrm_dataset/, enumerates checkpoint examples with
metadata labels, and provides splits for BTRM dual-head training (scrimble
for attention backend quality, scrongle for step count quality).

No pairing logic -- just enumerate all examples with provenance metadata.
Pairing is done combinatorially at consumption time via bt_loss_allpairs.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

import torch
from torch import Tensor

from .sampling import build_sigmas, simple_scheduler


@dataclass
class TrajectoryExample:
    """A single scorable example: one checkpoint from one trajectory."""

    traj_dir: str           # path to traj_NNNNNN/
    step_key: str           # "step_00", "step_04", ..., "final"
    step_idx: int           # 0, 4, 9, ...  (-1 for final)
    sigma: float            # sigma at this step (from trajectory's n_steps schedule)
    prompt: str             # text prompt
    prompt_idx: int         # prompt template index (-1 for i2i)
    attn_backend: str       # "sdpa" or "sage"
    n_steps: int            # trajectory step count
    traj_type: str          # "t2i" or "i2i"
    seed: int


class TrajectoryPool:
    """Load manifest and build a flat list of scorable examples."""

    def __init__(self, dataset_dir: str, include_i2i: bool = False):
        """Load manifest.json and build example list.

        Args:
            dataset_dir: Path to btrm_dataset/ (WSL or Windows path).
            include_i2i: If True, include i2i trajectories (non-batchable).
        """
        manifest_path = os.path.join(dataset_dir, "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)

        self._examples: list[TrajectoryExample] = []
        self._sigma_cache: dict[int, Tensor] = {}  # n_steps -> sigmas

        for record in manifest["records"]:
            traj_type = record["type"]
            if traj_type == "i2i" and not include_i2i:
                continue

            n_steps = record["n_steps"]
            sigmas = self._get_sigmas(n_steps)

            # Use traj_dir as-is (Windows paths when running Windows Python,
            # WSL paths when running Linux Python)
            traj_dir = record["traj_dir"]
            if sys.platform != "win32":
                traj_dir = self._to_wsl_path(traj_dir)

            if not os.path.isdir(traj_dir):
                continue

            prompt_idx = record.get("prompt_idx", -1)
            attn_backend = record["precision"]  # "sdpa" or "sage"

            # Enumerate checkpoints that exist on disk
            for fname in sorted(os.listdir(traj_dir)):
                if not fname.endswith(".pt"):
                    continue

                if fname == "final.pt":
                    step_key = "final"
                    step_idx = -1
                    sigma = 0.0  # final latent is after last step
                elif fname.startswith("step_"):
                    step_key = fname[:-3]  # "step_04"
                    step_idx = int(step_key.split("_")[1])
                    if step_idx >= n_steps:
                        continue  # skip checkpoints beyond trajectory length
                    sigma = float(sigmas[step_idx])
                else:
                    continue

                self._examples.append(TrajectoryExample(
                    traj_dir=traj_dir,
                    step_key=step_key,
                    step_idx=step_idx,
                    sigma=sigma,
                    prompt=record["prompt"],
                    prompt_idx=prompt_idx,
                    attn_backend=attn_backend,
                    n_steps=n_steps,
                    traj_type=traj_type,
                    seed=record["seed"],
                ))

    @property
    def examples(self) -> list[TrajectoryExample]:
        return self._examples

    def load_checkpoint(self, example: TrajectoryExample) -> Tensor:
        """Load the .pt file for this example."""
        path = os.path.join(example.traj_dir, f"{example.step_key}.pt")
        return torch.load(path, map_location="cpu", weights_only=True)

    def scrimble_split(self) -> tuple[list[int], list[int]]:
        """Return (sdpa_indices, sage_indices) for head 0 training.

        Only includes 30-step t2i trajectories (batches 0 and 1).
        Excludes final checkpoints (sigma=0 is not useful for scoring).
        """
        sdpa_indices = []
        sage_indices = []
        for i, ex in enumerate(self._examples):
            if ex.traj_type != "t2i" or ex.n_steps != 30:
                continue
            if ex.step_idx == -1:  # skip final
                continue
            if ex.attn_backend == "sdpa":
                sdpa_indices.append(i)
            else:
                sage_indices.append(i)
        return sdpa_indices, sage_indices

    def scrongle_split(self) -> tuple[list[int], list[int]]:
        """Return (full_step_indices, reduced_step_indices) for head 1.

        Full = 30 steps, reduced = <30 steps. Both t2i only.
        Excludes final checkpoints.
        """
        full_indices = []
        reduced_indices = []
        for i, ex in enumerate(self._examples):
            if ex.traj_type != "t2i":
                continue
            if ex.step_idx == -1:  # skip final
                continue
            if ex.n_steps == 30:
                full_indices.append(i)
            else:
                reduced_indices.append(i)
        return full_indices, reduced_indices

    def _get_sigmas(self, n_steps: int) -> Tensor:
        """Get sigma schedule for a given step count (cached)."""
        if n_steps not in self._sigma_cache:
            sigma_table = build_sigmas(shift=1.0, multiplier=1000.0)
            self._sigma_cache[n_steps] = simple_scheduler(sigma_table, n_steps)
        return self._sigma_cache[n_steps]

    @staticmethod
    def _to_wsl_path(win_path: str) -> str:
        """Convert Windows path (F:\\foo\\bar) to WSL path (/mnt/f/foo/bar)."""
        if not (len(win_path) >= 2 and win_path[1] == ":"):
            return win_path  # already a unix path
        drive = win_path[0].lower()
        rest = win_path[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"

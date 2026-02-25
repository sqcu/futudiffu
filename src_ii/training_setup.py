"""Composable setup phases for BTRM training scripts.

Replaces 200-300 lines of identical lifecycle code duplicated across
scripts_ii/run_reward_validated_training.py, run_flops_budget_100step_v2.py,
run_pinkify_validated_training.py, etc.

Each function covers one lifecycle phase:
  1. encode_training_prompts — load TE, encode, free TE
  2. build_dataset_positions — read V2 dataset, build positions + sampler
  3. load_training_backbone — load ZImageRLAIF, install LoRA, create optimizer
  4. load_latent_fn built via dataset_io.make_load_latent_fn

Import constraints:
  - futudiffu.text_encoder for TE lifecycle
  - futudiffu.dataset_v2 for DatasetReader
  - src_ii.pair_sampler for positions + sampler
  - src_ii.zimage_model for model loading
  - src_ii.btrm_lifecycle for training setup
  - src_ii.dataset_io for load_latent_fn factory
  - No server/client imports
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor


def encode_training_prompts(
    reader,
    traj_ids: Sequence[int],
    tokenizer_path: str,
    te_path: str,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, Tensor]:
    """Load text encoder, encode all unique prompts, free TE.

    Follows the mandatory lifecycle: load TE -> encode -> del TE -> empty_cache.
    TE and backbone must NEVER be loaded simultaneously (~8GB each).

    Returns:
        prompt_cache: dict mapping prompt string -> (1, cap_len, 2560) tensor on CPU.
    """
    from futudiffu.text_encoder import create_tokenizer, load_text_encoder, encode_prompt

    tokenizer = create_tokenizer(tokenizer_path)
    te_model = load_text_encoder(te_path, device=device, dtype=dtype)

    prompt_cache: dict[str, Tensor] = {}
    for idx in traj_ids:
        meta, _ = reader[idx]
        prompt = meta.get("prompt", "")
        if prompt and prompt not in prompt_cache:
            cond = encode_prompt(te_model, tokenizer, prompt, device=device)
            prompt_cache[prompt] = cond.cpu()

    n_prompts = len(prompt_cache)
    print(f"[training_setup] Encoded {n_prompts} unique prompts")

    del te_model, tokenizer
    torch.cuda.empty_cache()
    print(f"[training_setup] TE freed. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    return prompt_cache


def build_dataset_positions(
    dataset_dir: str | Path,
    traj_ids: Sequence[int] | None = None,
    clean_fraction: float = 0.8,
    rng_seed: int = 42,
    allow_inter_trajectory: bool = True,
    allow_intra_trajectory: bool = True,
    flops_weighted: bool = True,
):
    """Read V2 dataset, build positions, create pair sampler.

    Args:
        dataset_dir: Path to V2 dataset directory.
        traj_ids: Trajectory IDs to use. None = all available.
        clean_fraction: Fraction of clean-biased sampling (sigma=0).
        rng_seed: RNG seed for reproducibility.
        allow_inter_trajectory: Allow pairs across trajectories.
        allow_intra_trajectory: Allow pairs within a trajectory.
        flops_weighted: Compute FLOPS-based sampling weights.

    Returns:
        (reader, positions, sampler, traj_ids)
    """
    from futudiffu.dataset_v2 import DatasetReader
    from src_ii.pair_sampler import BTRMPairSampler, build_positions_from_v2

    reader = DatasetReader(str(dataset_dir))
    n_available = len(reader)
    print(f"[training_setup] Dataset: {n_available} trajectories")

    if traj_ids is None:
        traj_ids = list(range(n_available))

    positions = build_positions_from_v2(reader, traj_ids=traj_ids)
    print(f"[training_setup] Positions: {len(positions)} across {len(traj_ids)} trajectories")

    flops_weights = None
    if flops_weighted:
        from src_ii.flops_sampling import compute_flops_sampling_weights_from_positions
        flops_weights = compute_flops_sampling_weights_from_positions(positions)

    sampler = BTRMPairSampler(
        positions=positions,
        allow_inter_trajectory=allow_inter_trajectory,
        allow_intra_trajectory=allow_intra_trajectory,
        rng_seed=rng_seed,
        flops_weights=flops_weights,
        clean_fraction=clean_fraction,
    )

    return reader, positions, sampler, traj_ids


def load_training_backbone(
    fp8_path: str,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    adapter_name: str = "rtheta",
    adapter_rank: int = 8,
    adapter_alpha: float = 16.0,
    adapter_init_b_std: float = 0.01,
    lr: float = 3e-4,
    weight_decay: float = 0.0,
    betas: tuple[float, float] = (0.9, 0.999),
    optimizer_type: str = "adam",
    gradient_checkpointing: bool = True,
    btrm_checkpoint_dir: str | Path | None = None,
) -> tuple:
    """Load ZImageRLAIF, install LoRA, create optimizer, optionally load checkpoint.

    Args:
        fp8_path: Path to FP8 blockwise weights.
        device: Target device.
        dtype: Target dtype for non-FP8 params.
        adapter_name: LoRA adapter name.
        adapter_rank: LoRA rank.
        adapter_alpha: LoRA alpha.
        adapter_init_b_std: B matrix init std (0.01 = nonzero gradient).
        lr: Learning rate.
        weight_decay: Weight decay.
        betas: Adam betas.
        optimizer_type: "adam" or "muon".
        gradient_checkpointing: Enable gradient checkpointing.
        btrm_checkpoint_dir: If set, load adapter + head from this directory.

    Returns:
        (raw_model, optimizer, head_names)
    """
    from src_ii.zimage_model import load_zimage_rlaif
    from src_ii.btrm_lifecycle import setup_btrm_training, load_btrm

    raw_model = load_zimage_rlaif(
        fp8_path, device=device, dtype=dtype,
        compile_model=False, fuse=True,
    )
    print(f"[training_setup] VRAM after backbone: "
          f"{torch.cuda.memory_allocated() / 1e9:.2f} GB")

    optimizer = setup_btrm_training(
        raw_model,
        adapter_name=adapter_name,
        adapter_rank=adapter_rank,
        adapter_alpha=adapter_alpha,
        adapter_init_b_std=adapter_init_b_std,
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        optimizer_type=optimizer_type,
        gradient_checkpointing=gradient_checkpointing,
    )

    if btrm_checkpoint_dir is not None:
        load_btrm(raw_model, adapter_name, btrm_checkpoint_dir)
        print(f"[training_setup] Loaded BTRM checkpoint from {btrm_checkpoint_dir}")

    head_names = []
    n_heads = raw_model.n_score_heads
    default_names = ["bit_quality", "step_quality"]
    for i in range(n_heads):
        head_names.append(default_names[i] if i < len(default_names) else f"head_{i}")

    from src_ii.multi_lora import get_adapter_params
    n_adapter = sum(p.numel() for p in get_adapter_params(raw_model, adapter_name).values())
    n_head = sum(p.numel() for p in raw_model.score_proj.parameters()) + \
             sum(p.numel() for p in raw_model.score_norm.parameters())
    print(f"[training_setup] Adapter params: {n_adapter:,}")
    print(f"[training_setup] Head params: {n_head:,}")
    print(f"[training_setup] Total trainable: {n_adapter + n_head:,}")

    return raw_model, optimizer, head_names

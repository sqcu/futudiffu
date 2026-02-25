"""Multi-GPU data-parallel gradient synchronization.

Provides the minimal distributed primitives needed for BTRM training across
multiple GPUs. Each GPU runs the full training loop with its own data. Gradients
are all-reduced before optimizer.step().

Usage:
    rank, world_size, is_dist = setup_distributed()
    sync_fn = make_gradient_sync_fn(trainable_params, world_size) if is_dist else None
    ...
    # In training loop, after backward:
    if sync_fn is not None:
        sync_fn()
    optimizer.step()

Environment variables (set by torchrun/torch.distributed.launch):
    RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn


def setup_distributed() -> tuple[int, int, bool]:
    """Initialize distributed process group if environment is set up.

    Returns:
        (rank, world_size, is_distributed)
        If not in a distributed context, returns (0, 1, False).
    """
    import os

    if "RANK" not in os.environ:
        return 0, 1, False

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    print(f"[distributed] rank={rank}, world_size={world_size}, "
          f"local_rank={local_rank}, device=cuda:{local_rank}")
    return rank, world_size, True


def make_gradient_sync_fn(
    params: Sequence[nn.Parameter],
    world_size: int,
) -> Callable[[], None]:
    """Build a closure that all-reduces gradients across ranks.

    The returned callable divides each param's .grad by world_size
    (mean reduction) and synchronizes via all_reduce.

    Args:
        params: Trainable parameters to sync.
        world_size: Number of ranks.

    Returns:
        Callable that performs in-place gradient synchronization.
    """
    params = [p for p in params if p.requires_grad]

    def sync_gradients():
        for p in params:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad.div_(world_size)

    return sync_gradients


def is_rank_zero() -> bool:
    """True if this is rank 0 or not in a distributed context."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def barrier():
    """Synchronization barrier. No-op if not distributed."""
    if dist.is_initialized():
        dist.barrier()


def cleanup():
    """Destroy the process group. Call at end of training."""
    if dist.is_initialized():
        dist.destroy_process_group()

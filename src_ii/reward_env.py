"""BTRM reward environment protocol for DDGRPO policy optimization.

Maps multi-head BTRM scores → scalar rewards → group-relative advantages.
Pure math — imports only torch + ddreinforce.group_advantages.

Three target environments:
  - PINKIFY: single BTRM head (pinkify quality)
  - TNT: single BTRM head (thisnotthat structural similarity)
  - PINKIFY × TNT: pinkify_reward * group_advantages(tnt_rewards)

Multiple concurrent policy runs share one server via multi-LoRA adapter
routing. Each PolicyConfig binds an adapter name to an environment.

Import constraints:
  - torch only (+ ddreinforce.group_advantages from this package)
  - No futudiffu imports
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from .ddreinforce import group_advantages


# ---------------------------------------------------------------------------
# Reward environment base
# ---------------------------------------------------------------------------

class BTRMRewardEnv:
    """Base reward environment. Subclasses implement compute_rewards."""

    name: str

    def compute_rewards(self, scores: Tensor) -> Tensor:
        """Map (K, n_heads) BTRM scores to (K,) scalar rewards."""
        raise NotImplementedError

    def compute_advantages(self, scores: Tensor) -> tuple[Tensor, Tensor]:
        """Compute rewards and group-relative advantages.

        Args:
            scores: (K, n_heads) BTRM scores for K trajectories.

        Returns:
            (rewards, advantages) each (K,).
        """
        rewards = self.compute_rewards(scores)
        advantages = group_advantages(rewards)
        return rewards, advantages

    def reward_metadata(self) -> dict:
        """Return metadata about this environment for logging."""
        return {"env_name": self.name, "env_type": type(self).__name__}


# ---------------------------------------------------------------------------
# Single-head environment
# ---------------------------------------------------------------------------

class SingleHeadRewardEnv(BTRMRewardEnv):
    """One BTRM head column as scalar reward.

    Use for PINKIFY-only or TNT-only policy training.
    """

    def __init__(self, name: str, head_idx: int, head_name: str):
        self.name = name
        self.head_idx = head_idx
        self.head_name = head_name

    def compute_rewards(self, scores: Tensor) -> Tensor:
        return scores[:, self.head_idx]

    def reward_metadata(self) -> dict:
        return {
            "env_name": self.name,
            "env_type": "SingleHeadRewardEnv",
            "head_idx": self.head_idx,
            "head_name": self.head_name,
        }


# ---------------------------------------------------------------------------
# Composed environment: LHS_reward * group_advantages(RHS)
# ---------------------------------------------------------------------------

class ComposedRewardEnv(BTRMRewardEnv):
    """Composed reward: lhs_reward * group_advantages(rhs_scores).

    The LHS head provides the primary reward signal. The RHS head
    modulates it through group-relative advantages — trajectories
    that are also good on the RHS axis get amplified, those that
    are bad get attenuated.

    This is the PINKIFY × TNT environment: optimize pinkify quality
    while respecting structural similarity constraints.
    """

    def __init__(
        self,
        name: str,
        lhs_head_idx: int,
        rhs_head_idx: int,
        lhs_name: str,
        rhs_name: str,
    ):
        self.name = name
        self.lhs_head_idx = lhs_head_idx
        self.rhs_head_idx = rhs_head_idx
        self.lhs_name = lhs_name
        self.rhs_name = rhs_name

    def compute_rewards(self, scores: Tensor) -> Tensor:
        lhs = scores[:, self.lhs_head_idx]
        rhs_adv = group_advantages(scores[:, self.rhs_head_idx])
        return lhs * rhs_adv

    def reward_metadata(self) -> dict:
        return {
            "env_name": self.name,
            "env_type": "ComposedRewardEnv",
            "lhs_head_idx": self.lhs_head_idx,
            "lhs_name": self.lhs_name,
            "rhs_head_idx": self.rhs_head_idx,
            "rhs_name": self.rhs_name,
        }


# ---------------------------------------------------------------------------
# Policy configuration
# ---------------------------------------------------------------------------

@dataclass
class PolicyConfig:
    """Configuration for one DDGRPO policy run.

    Binds an adapter name to a reward environment + hyperparameters.
    """
    adapter_name: str
    env: BTRMRewardEnv
    lr: float = 1e-4
    beta: float = 0.04
    clip_eps: float = 0.2
    eta_scale: float = 0.1
    K: int = 4
    n_steps: int = 30
    max_grad_norm: float = 1.0
    adapter_rank: int = 8
    adapter_alpha: float = 16.0
    init_b_std: float = 0.01
    width: int = 1280
    height: int = 832
    cfg: float = 4.0


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_standard_envs(
    head_names: tuple[str, ...],
) -> dict[str, BTRMRewardEnv]:
    """Create the three standard reward environments.

    Args:
        head_names: BTRM head names in order. Must contain "pinkify" and
            "thisnotthat" — raises ValueError if either is missing.

    Returns:
        {"pinkify": SingleHead, "tnt": SingleHead, "pinkify_x_tnt": Composed}
    """
    pinkify_idx = head_names.index("pinkify")
    tnt_idx = head_names.index("thisnotthat")

    return {
        "pinkify": SingleHeadRewardEnv("pinkify", pinkify_idx, "pinkify"),
        "tnt": SingleHeadRewardEnv("tnt", tnt_idx, "thisnotthat"),
        "pinkify_x_tnt": ComposedRewardEnv(
            "pinkify_x_tnt",
            lhs_head_idx=pinkify_idx,
            rhs_head_idx=tnt_idx,
            lhs_name="pinkify",
            rhs_name="thisnotthat",
        ),
    }

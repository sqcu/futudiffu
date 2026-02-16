"""Multi-head Bradley-Terry Reward Model (BTRM) for diffusion QAT.

Scores NextDiT hidden states to provide reward signals for REINFORCE-based
quantization-aware training. Architecture adapted from dialogue_yoinker's
multihead BTRM spec (2025-12-27-btrm-multihead-spec.md).

Pipeline:
    NextDiT backbone (frozen) -> final_layer hidden states (B, N_tokens, 3840)
      -> mean pool over token dim -> (B, 3840)
      -> RMSNorm(3840)
      -> Linear(3840, N_heads, bias=False)
      -> soft_tanh_cap
      -> (B, N_heads) scalar scores

The BTRM head parameters are SEPARATE from the diffusion model and any LoRA
parameters. The backbone is always frozen during BTRM inference.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .attention import rms_norm
from .diffusion_model import NextDiT


# ---------------------------------------------------------------------------
# Default configuration for diffusion QAT
# ---------------------------------------------------------------------------

DEFAULT_HEADS: tuple[str, ...] = ("bit_quality", "step_quality")

DEFAULT_TIER_WEIGHTS: dict[str, float] = {
    "soft_neg": 2.0,       # close to full-precision (e.g., FP8+BF16)
    "semi_firm_neg": 1.0,  # moderately degraded (e.g., INT8 or FP8+FP8)
    "furthest_neg": 0.5,   # heavily degraded (e.g., FP8+FP8 at low steps)
}


# ---------------------------------------------------------------------------
# RMSNorm layer (learnable, using futudiffu.attention.rms_norm kernel)
# ---------------------------------------------------------------------------

class _RMSNorm(nn.Module):
    """Learnable RMSNorm that delegates to the same kernel as the rest of futudiffu."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return rms_norm(x, self.weight, self.eps)


# ---------------------------------------------------------------------------
# BTRM scoring head
# ---------------------------------------------------------------------------

class BTRMHead(nn.Module):
    """Multi-head Bradley-Terry reward scoring head.

    Takes pooled hidden states from a frozen NextDiT backbone and produces
    per-head scalar scores.

    Args:
        hidden_dim: Dimension of the backbone hidden states (3840 for NextDiT Z-Image).
        head_names: Names for each scoring head.
        logit_cap: Soft tanh cap magnitude. Scores are bounded to +/-logit_cap.
                   Set to 0.0 to disable capping.
    """

    def __init__(
        self,
        hidden_dim: int = 3840,
        head_names: Sequence[str] = DEFAULT_HEADS,
        logit_cap: float = 10.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_names = tuple(head_names)
        self.n_heads = len(self.head_names)
        self.logit_cap = logit_cap

        self.norm = _RMSNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, self.n_heads, bias=False)

        # Initialize near zero so initial scores are small
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Score hidden states.

        Args:
            hidden_states: (B, N_tokens, hidden_dim) from the final transformer
                block of NextDiT (before the output projection / final_layer).

        Returns:
            (B, N_heads) scalar scores, one per head.
        """
        # Cast input to match model param dtype (handles bf16 server -> float32 head)
        param_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.to(dtype=param_dtype)

        # Mean pool over token dimension
        pooled = hidden_states.mean(dim=1)  # (B, hidden_dim)

        # Normalize then project
        normed = self.norm(pooled)           # (B, hidden_dim)
        scores = self.proj(normed)           # (B, n_heads)

        # Soft tanh capping: smooth bounded scores, no graph breaks
        if self.logit_cap > 0:
            scores = self.logit_cap * torch.tanh(scores / self.logit_cap)

        return scores

    def get_head_idx(self, head_name: str) -> int:
        """Return the integer index for a named head."""
        return self.head_names.index(head_name)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def bradley_terry_loss(pos_scores: Tensor, neg_scores: Tensor) -> Tensor:
    """Standard Bradley-Terry ranking loss.

    P(pos > neg) = sigmoid(pos - neg)
    Loss = -log_sigmoid(pos - neg).mean()

    Args:
        pos_scores: (N,) scores for positive samples.
        neg_scores: (N,) scores for negative samples (same length).

    Returns:
        Scalar loss.
    """
    return -F.logsigmoid(pos_scores - neg_scores).mean()


def bt_loss_allpairs(pos_scores: Tensor, neg_scores: Tensor) -> Tensor:
    """BT loss over all (pos, neg) combinations.

    Unlike bradley_terry_loss which requires matched-length inputs,
    this computes loss over the full n_pos x n_neg cross-product.

    Args:
        pos_scores: (n_pos,) scores that should be higher.
        neg_scores: (n_neg,) scores that should be lower.

    Returns:
        Scalar loss: mean -log_sigmoid(pos_i - neg_j) over all i,j.
    """
    diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
    return -F.logsigmoid(diff).mean()


def logsquare_regularizer(scores: Tensor, eps: float = 1e-6) -> Tensor:
    """Logsquare regularization: pushes positive scores toward r ~ 1.

    log(r^2 + eps).mean()

    At r=1: log(1) = 0 (optimal).
    At r=0.1: log(0.01) ~ -4.6 (penalized).
    At r=10: log(100) ~ 4.6 (penalized).

    This is NOT MSE to batch mean.

    Args:
        scores: (N,) scores to regularize.
        eps: Small constant for numerical stability.

    Returns:
        Scalar regularization term.
    """
    return torch.log(scores ** 2 + eps).mean()


def compute_multihead_loss(
    pos_scores: Tensor,
    neg_scores_by_tier: dict[str, Tensor | None],
    pos_head_indices: Tensor,
    head_names: Sequence[str],
    tier_weights: dict[str, float] | None = None,
    logsquare_weight: float = 0.1,
) -> dict[str, Tensor | float]:
    """Compute multi-head BT loss with per-head masking and negative tiers.

    Each positive sample belongs to exactly one head (indicated by
    ``pos_head_indices``). For each head that has positives in the batch:

    1. BT loss is computed against every negative tier (weighted).
    2. Logsquare regularization is applied to that head's positive scores.

    Losses are averaged across active heads.

    Args:
        pos_scores: (n_pos, n_heads) scores for all positive samples.
        neg_scores_by_tier: Mapping from tier name (e.g. "soft_neg") to
            (n_neg, n_heads) tensor of negative scores, or None to skip.
        pos_head_indices: (n_pos,) integer tensor mapping each positive to
            its head index.
        head_names: Ordered head names (for diagnostics).
        tier_weights: Per-tier scalar weights. Defaults to DEFAULT_TIER_WEIGHTS.
        logsquare_weight: Weight for logsquare regularization term.

    Returns:
        Dictionary with keys:
            "loss"       - total scalar loss
            "bt_loss"    - BT component (before weighting)
            "logsq_loss" - logsquare component (before weighting)
            "active_heads" - number of heads with positives in this batch
            "per_head_bt" - dict mapping head_name -> per-head BT loss (detached)
    """
    if tier_weights is None:
        tier_weights = DEFAULT_TIER_WEIGHTS

    n_heads = pos_scores.size(1)

    total_bt = pos_scores.new_zeros(())
    total_logsq = pos_scores.new_zeros(())
    active_heads = 0
    per_head_bt: dict[str, float] = {}

    for head_idx in range(n_heads):
        head_mask = pos_head_indices == head_idx
        n_head_pos = head_mask.sum().item()

        if n_head_pos == 0:
            continue

        active_heads += 1
        head_pos = pos_scores[head_mask, head_idx]  # (n_head_pos,)

        # BT loss against each negative tier
        head_bt = pos_scores.new_zeros(())
        for tier_name, neg_scores in neg_scores_by_tier.items():
            if neg_scores is None:
                continue
            head_neg = neg_scores[:, head_idx]  # (n_neg,)
            n_neg = head_neg.size(0)
            if n_neg == 0:
                continue

            weight = tier_weights.get(tier_name, 1.0)

            # Match batch sizes for pairwise comparison
            if n_neg >= n_head_pos:
                bt = bradley_terry_loss(head_pos, head_neg[:n_head_pos])
            else:
                bt = bradley_terry_loss(head_pos[:n_neg], head_neg)

            head_bt = head_bt + weight * bt

        total_bt = total_bt + head_bt
        per_head_bt[head_names[head_idx]] = head_bt.detach().item()

        # Logsquare regularization on positives only
        if logsquare_weight > 0:
            total_logsq = total_logsq + logsquare_regularizer(head_pos)

    # Average across active heads
    if active_heads > 0:
        total_bt = total_bt / active_heads
        total_logsq = total_logsq / active_heads

    total_loss = total_bt + logsquare_weight * total_logsq

    return {
        "loss": total_loss,
        "bt_loss": total_bt,
        "logsq_loss": total_logsq,
        "active_heads": active_heads,
        "per_head_bt": per_head_bt,
    }


def compute_labeled_btrm_loss(
    all_scores: Tensor,
    labels: list[dict],
    head_names: Sequence[str],
    logsquare_weight: float = 0.1,
) -> dict[str, Tensor | float]:
    """Compute multi-head BT loss from flat per-example labels.

    Each label dict has ``head_idx`` (int) and ``is_positive`` (bool).
    Uses all-pairs BT loss within each head (cross-product of positives
    and negatives) plus logsquare regularization on positives.

    Args:
        all_scores: (N, n_heads) scores from the BTRM head.
        labels: Length-N list of dicts with head_idx, is_positive.
        head_names: Ordered head names (for accuracy dict keys).
        logsquare_weight: Weight for logsquare regularization.

    Returns:
        Dictionary with keys: loss, bt_loss, logsq_loss,
        per_head_accuracy, active_heads.
    """
    n_heads = all_scores.size(1)
    device = all_scores.device

    total_bt = all_scores.new_zeros(())
    total_logsq = all_scores.new_zeros(())
    active_heads = 0
    per_head_accuracy: dict[str, float] = {}

    for head_idx in range(n_heads):
        pos_mask = torch.tensor(
            [l["head_idx"] == head_idx and l["is_positive"] for l in labels],
            device=device,
        )
        neg_mask = torch.tensor(
            [l["head_idx"] == head_idx and not l["is_positive"] for l in labels],
            device=device,
        )

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue

        active_heads += 1
        pos_scores = all_scores[pos_mask, head_idx]
        neg_scores = all_scores[neg_mask, head_idx]

        bt = bt_loss_allpairs(pos_scores, neg_scores)
        total_bt = total_bt + bt

        with torch.no_grad():
            diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
            acc = (diff > 0).float().mean().item()
            per_head_accuracy[head_names[head_idx]] = acc

        if logsquare_weight > 0:
            total_logsq = total_logsq + logsquare_regularizer(pos_scores)

    if active_heads > 0:
        total_bt = total_bt / active_heads
        total_logsq = total_logsq / active_heads

    total_loss = total_bt + logsquare_weight * total_logsq

    return {
        "loss": total_loss,
        "bt_loss": total_bt,
        "logsq_loss": total_logsq,
        "per_head_accuracy": per_head_accuracy,
        "active_heads": active_heads,
    }


# ---------------------------------------------------------------------------
# Wrapper: frozen NextDiT backbone + BTRM head
# ---------------------------------------------------------------------------

class BTRMWrapper(nn.Module):
    """Combines a frozen NextDiT backbone with a trainable BTRMHead.

    The wrapper hooks into NextDiT's forward pass to capture the hidden
    states from the last transformer block (before ``final_layer``), pools
    them, and routes them through the BTRM head.

    The backbone parameters are always frozen. Only the BTRMHead trains.

    Args:
        model: A NextDiT instance (will be set to eval and frozen).
        head: A BTRMHead instance. If None, one is created with default config
              using the model's hidden dimension.
    """

    def __init__(self, model: NextDiT, head: BTRMHead | None = None) -> None:
        super().__init__()
        self.model = model
        self.head = head if head is not None else BTRMHead(hidden_dim=model.dim)

        # Freeze backbone
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Storage for the hook capture
        self._captured_hidden: Tensor | None = None
        self._hook_handle = None

    def _install_hook(self) -> None:
        """Install a forward hook on the last transformer block to capture
        hidden states before they reach final_layer."""
        if self._hook_handle is not None:
            return  # Already installed

        # The last element of model.layers is the final JointTransformerBlock.
        # Its output is the (B, N_tokens, dim) tensor that gets passed to
        # final_layer. We capture it via a forward hook.
        last_block = self.model.layers[-1]

        def _capture_hook(_module: nn.Module, _input: tuple, output: Tensor) -> None:
            self._captured_hidden = output

        self._hook_handle = last_block.register_forward_hook(_capture_hook)

    def _remove_hook(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    @torch.no_grad()
    def _run_backbone(
        self,
        x: Tensor,
        timesteps: Tensor,
        context: Tensor,
        num_tokens: int,
        attention_mask: Tensor | None = None,
        rope_cache: dict | None = None,
    ) -> Tensor:
        """Run the frozen backbone and return captured hidden states.

        Returns:
            (B, N_tokens, dim) hidden states from the final transformer block.
        """
        self._install_hook()
        self._captured_hidden = None

        # Forward pass through the full model (output is discarded)
        self.model(
            x, timesteps, context, num_tokens,
            attention_mask=attention_mask,
            rope_cache=rope_cache,
        )

        assert self._captured_hidden is not None, (
            "Hook failed to capture hidden states. "
            "Ensure the model has at least one layer in model.layers."
        )
        hidden = self._captured_hidden
        self._captured_hidden = None
        return hidden

    def score(
        self,
        x: Tensor,
        timesteps: Tensor,
        context: Tensor,
        num_tokens: int,
        attention_mask: Tensor | None = None,
        rope_cache: dict | None = None,
    ) -> Tensor:
        """Score a batch of diffusion model inputs.

        Runs the frozen backbone, extracts hidden states from the final
        transformer block, and passes them through the BTRM head.

        Args:
            x: (B, C, H, W) noisy latent.
            timesteps: (B,) sigma values.
            context: (B, seq, cap_feat_dim) text encoder hidden states.
            num_tokens: Number of text tokens.
            attention_mask: Optional attention mask for text.
            rope_cache: Optional precomputed RoPE cache.

        Returns:
            (B, N_heads) reward scores.
        """
        hidden = self._run_backbone(
            x, timesteps, context, num_tokens,
            attention_mask=attention_mask,
            rope_cache=rope_cache,
        )
        # hidden: (B, N_tokens, dim) -- includes both caption and image tokens
        return self.head(hidden)

    def train(self, mode: bool = True) -> BTRMWrapper:
        """Only the head trains; backbone stays frozen in eval."""
        self.head.train(mode)
        # Backbone is always eval
        self.model.eval()
        return self

    def eval(self) -> BTRMWrapper:
        self.head.eval()
        self.model.eval()
        return self

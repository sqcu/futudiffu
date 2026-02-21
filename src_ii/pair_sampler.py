"""On-the-fly BTRM pair sampler for V2 datasets.

Replaces the materialized pair table (280 fixed pairs from 10 trajectories)
with stochastic sampling over the full combinatorial space. With 259+
trajectories in V2, the pair space is ~1.6M+ pairs. Materializing them all
is wasteful and inflexible.

Core principle: EVERY image has a pinkify score and a thisnotthat score.
Any two images can form a valid pairwise comparison. Pinkify discriminates
attention quantization artifacts (SageAttention INT8 QK vs SDPA) -- a
property of the rendering backend, not the content or resolution. Thisnotthat
discriminates step count (30-step vs 8-22 step) -- also backend metadata.
These are UNIVERSAL scoring functions. There is no requirement that paired
images share a prompt, resolution, trajectory, or any other content property.
The only requirement for a valid pair is that we know the ground-truth
preference ordering (which image should score higher on each head).

Instead of restricting pair formation, this module:
  1. Indexes all (trajectory, step) positions from the V2 dataset
  2. Assigns logSNR-based sampling weights to each position
  3. Samples pairs on-the-fly using two-stage sampling:
     a. Pick trajectory (uniform or weighted)
     b. Pick step within trajectory (weighted by geometric logSNR decay)

Preference computation is NOT this module's job. The sampler returns
(image_a, image_b) metadata; the training loop evaluates preferences
via a separate preference_fn. This decoupling allows swapping between
BTRM tasks (scrimblo/scrongle, PINKER/TNT) without changing the pair
selection algorithm.

The pair distribution is shaped by logSNR weighting: cleaner images
(higher logSNR, lower sigma) get full weight, while noisier images
get geometrically decaying weight. The decay schedule is parameterized
by (threshold, interval, decay_rate) per user spec:
  "noisy latents with log(snr(t)) > 10 get sampled uniformly, every -5
  step decrement below log(snr(t)) 10 gets p-% geometric decay."
  - Default: threshold=10.0, interval=5.0, decay_rate=0.5
  - Gentle: threshold=10.0, interval=5.0, decay_rate=0.75
sigma=0 (fully denoised) gets FULL weight -- it is the most important
training signal for the reward model.

6-tier resolution bucketing: trajectories are classified into 6 tiers
via assign_budget_tier() from resolution_sampling.py. The stratified
batch sampler allocates pair slots across all populated tiers to
approximate a target FLOPS distribution. Cross-resolution pairs are
formed by sampling image A from one tier and image B from any other
tier -- no prompt matching required.

Performance: pure Python/numpy for sampling logic. No GPU ops, no torch
ops in the sampling hot path. The only torch dependency is sigma schedule
construction at init time.

Import constraints:
  - math, random for sampling
  - numpy for softmax/Gumbel-max
  - torch only at init time (sigma schedule construction)
  - No futudiffu server/client imports
  - No reward function imports
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from src_ii.resolution_sampling import MEGAPIXEL_ANCHORS, assign_budget_tier
from src_ii.flops_sampling import flops_units, pair_flops_units


# ---------------------------------------------------------------------------
# LogSNR geometric decay sampling weight
# ---------------------------------------------------------------------------

def logsnr_sampling_weight(
    sigma: float,
    threshold: float = 10.0,
    interval: float = 5.0,
    decay_rate: float = 0.5,
) -> float:
    """Geometric decay sampling weight based on logSNR.

    User spec: "noisy latents with log(snr(t)) > 10 get sampled uniformly,
    every -5 step decrement below log(snr(t)) 10 gets p-% geometric decay."

    Flat at 1.0 for logSNR >= threshold, then decay_rate^((threshold - logSNR) / interval)
    below. This is a one-sided exponential ramp, NOT a sigmoid.

    sigma=0 (fully denoised, logSNR=+inf) gets FULL weight. This is the most
    important training signal for the reward model.

    The schedule is tunable:
        threshold: logSNR value where decay begins (default 10.0)
        interval: logSNR nats per decay step (default 5.0)
        decay_rate: multiplicative factor per interval (default 0.5)

    Verification table (default params: threshold=10.0, interval=5.0, decay_rate=0.5):

        For 1280x832 (shift=1.0, 30-step schedule):
            sigma=0.000  logSNR=+inf   weight=1.000   (fully denoised)
            sigma=0.034  logSNR=+6.69  weight=0.632   (step_29, near-clean)
            sigma=0.200  logSNR=+2.77  weight=0.367   (step_24, low noise)
            sigma=0.367  logSNR=+1.09  weight=0.291   (step_19, mid)
            sigma=0.500  logSNR=0.00   weight=0.250   (step_15, equal)
            sigma=0.700  logSNR=-1.69  weight=0.198   (step_09, noisy)
            sigma=0.867  logSNR=-3.75  weight=0.149   (step_04, very noisy)
            sigma=1.000  logSNR=-inf   weight~0.000   (pure noise)

        For 256x256 (shift=4.03, 30-step schedule):
            sigma=0.000  logSNR=+inf   weight=1.000   (fully denoised)
            sigma=0.124  logSNR=+3.91  weight=0.430   (step_29, near-clean)
            sigma=0.502  logSNR=-0.02  weight=0.250   (step_24, near-equal)
            sigma=0.700  logSNR=-1.70  weight=0.198   (step_19, noisy)
            sigma=0.822  logSNR=-3.06  weight=0.164   (step_14, very noisy)
            sigma=0.964  logSNR=-6.54  weight=0.101   (step_04, near-pure-noise)
            sigma=1.000  logSNR=-inf   weight~0.000   (pure noise)

    Clean images (sigma<=0, logSNR=+inf): weight = 1.0
    Pure noise (sigma>=1, logSNR=-inf): weight -> 0 (geometric limit)

    Resolution awareness: The sigma at a given step index depends on the
    resolution's sigma schedule (via resolution_shift). For 256x256 with
    shift~4.0, step_29 has sigma=0.124 (logSNR=+3.9), far from clean.
    For 1280x832 with shift=1.0, step_29 has sigma=0.034 (logSNR=+6.7).
    The weighting correctly operates on sigma/logSNR, not step index.
    """
    if sigma <= 0.0:
        return 1.0  # clean image: full weight
    if sigma >= 1.0:
        # pure noise: many intervals below threshold
        # use a large but finite number of intervals to avoid inf
        return decay_rate ** 20  # effectively zero but not literally 0

    log_snr = 2.0 * math.log((1.0 - sigma) / sigma)

    if log_snr >= threshold:
        return 1.0

    n_intervals = (threshold - log_snr) / interval
    return decay_rate ** n_intervals


def logsnr_sampling_logit(
    sigma: float,
    threshold: float = 10.0,
    interval: float = 5.0,
    decay_rate: float = 0.5,
) -> float:
    """Log of the geometric decay sampling weight. For use in softmax/Gumbel sampling.

    Equivalent to: log(logsnr_sampling_weight(sigma, ...))
    Returns 0.0 for logSNR >= threshold, negative ramp below.

    Args:
        sigma: Noise level from the diffusion schedule (0 <= sigma <= 1).
        threshold: logSNR value where decay begins (default 10.0).
        interval: logSNR nats per decay step (default 5.0).
        decay_rate: multiplicative factor per interval (default 0.5).

    Returns:
        Scalar logit (0 or negative). Use softmax over all positions to get
        sampling probabilities.
    """
    if sigma <= 0.0:
        return 0.0  # log(1.0) = 0
    if sigma >= 1.0:
        return math.log(decay_rate) * 20  # large negative

    log_snr = 2.0 * math.log((1.0 - sigma) / sigma)

    if log_snr >= threshold:
        return 0.0

    n_intervals = (threshold - log_snr) / interval
    return math.log(decay_rate) * n_intervals


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1D array."""
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def _sigma_to_logsnr(sigma: float) -> float:
    """Convert sigma to logSNR using the CONST noise model.

    logSNR = 2 * ln((1 - sigma) / sigma).
    Returns +inf for sigma <= 0, -inf for sigma >= 1.
    """
    if sigma <= 0.0:
        return float("inf")
    if sigma >= 1.0:
        return float("-inf")
    return 2.0 * math.log((1.0 - sigma) / sigma)


def _clean_biased_step_logits(
    positions: list[_ImagePosition],
    decay_scale: float = 0.5,
) -> np.ndarray:
    """Compute exponential-decay logits for non-clean step selection.

    For the 20% of samples that are NOT clean (sigma=0), we want an
    exponential decay biased toward high logSNR (slightly noisy images).

    The decay is parameterized so that:
      - An image at logSNR=+5 is ~10x more likely than logSNR=0
      - An image at logSNR=+5 is ~100x more likely than logSNR=-5

    The logit for each position is: decay_scale * logSNR.
    With decay_scale=0.5:
      logit(logSNR=+5) - logit(logSNR=0) = 0.5 * 5 = 2.5, ratio = e^2.5 ~ 12.2
      logit(logSNR=+5) - logit(logSNR=-5) = 0.5 * 10 = 5.0, ratio = e^5.0 ~ 148

    sigma=0 positions are EXCLUDED from the non-clean pool (they are handled
    by the clean_fraction coin flip). Their logits are set to -inf so softmax
    gives them probability 0.

    Args:
        positions: List of _ImagePosition objects for one trajectory.
        decay_scale: Multiplier for logSNR in the logit. Higher values
            make the distribution more peaked toward near-clean images.
            Default 0.5 gives roughly 10x preference per 5 logSNR nats.

    Returns:
        1D numpy array of logits (same length as positions). Positions
        with sigma=0 get logit -inf. If ALL positions are sigma=0,
        all logits are -inf (caller must handle this edge case).
    """
    logits = np.full(len(positions), -np.inf, dtype=np.float64)
    for i, pos in enumerate(positions):
        if pos.sigma <= 0.0:
            # sigma=0 is handled by the clean_fraction path, not here
            continue
        if pos.sigma >= 1.0:
            # Pure noise: extremely unlikely to sample
            logits[i] = -50.0
            continue
        logsnr = 2.0 * math.log((1.0 - pos.sigma) / pos.sigma)
        logits[i] = decay_scale * logsnr
    return logits


# ---------------------------------------------------------------------------
# LogSNR histogram bins for distribution tracking
# ---------------------------------------------------------------------------

# Bin edges: [-inf,-5), [-5,-2), [-2,0), [0,2), [2,5), [5,8), [8,+inf), sigma=0
_LOGSNR_HIST_BINS: list[tuple[str, float, float]] = [
    ("[-inf,-5)", float("-inf"), -5.0),
    ("[-5,-2)",   -5.0,         -2.0),
    ("[-2,0)",    -2.0,          0.0),
    ("[0,2)",      0.0,          2.0),
    ("[2,5)",      2.0,          5.0),
    ("[5,8)",      5.0,          8.0),
    ("[8,+inf)",   8.0,  float("inf")),
]
_SIGMA_ZERO_BIN = "sigma=0"


def _logsnr_hist_bin(sigma: float) -> str:
    """Assign a sigma value to a logSNR histogram bin label."""
    if sigma <= 0.0:
        return _SIGMA_ZERO_BIN
    if sigma >= 1.0:
        return _LOGSNR_HIST_BINS[0][0]  # [-inf,-5) for pure noise
    logsnr = 2.0 * math.log((1.0 - sigma) / sigma)
    for label, lo, hi in _LOGSNR_HIST_BINS:
        if lo <= logsnr < hi:
            return label
    return _LOGSNR_HIST_BINS[-1][0]  # fallback to [8,+inf)


# ---------------------------------------------------------------------------
# Image position: a (trajectory, step) pair with metadata
# ---------------------------------------------------------------------------

class _ImagePosition:
    """A single image position in the dataset: one (traj_id, step_key) pair.

    prompt_idx is retained for informational/logging purposes only. It is NOT
    used for pairing logic. Every image in the universe has a pinkify score and
    a thisnotthat score. Any two images can form a valid pairwise comparison
    regardless of prompt, resolution, or trajectory. The only requirement is
    that we know the ground-truth preference ordering.
    """
    __slots__ = ("traj_id", "step_key", "sigma", "logit", "width", "height", "prompt_idx")

    def __init__(
        self,
        traj_id: int,
        step_key: str,
        sigma: float,
        logit: float,
        width: int = 1280,
        height: int = 832,
        prompt_idx: int = -1,
    ):
        self.traj_id = traj_id
        self.step_key = step_key
        self.sigma = sigma
        self.logit = logit
        self.width = width
        self.height = height
        self.prompt_idx = prompt_idx


# ---------------------------------------------------------------------------
# PairSpec: typed return value from sample_macrobatch
# ---------------------------------------------------------------------------

@dataclass
class PairSpec:
    """A sampled image pair with FLOPS cost metadata.

    Used as the return type from sample_macrobatch(). Each PairSpec
    contains the full position metadata for both images, plus the
    computed FLOPS cost and whether this is a cross-resolution pair.

    The preferred/rejected ordering is NOT determined here -- that is
    the preference_fn's responsibility. image_a and image_b are just
    the two positions; downstream code assigns preference.
    """
    image_a: _ImagePosition
    image_b: _ImagePosition
    flops_cost: float  # In 1024^2-equivalent units (sum of both images)
    cross_resolution: bool  # True if image_a and image_b have different resolutions

    def to_pair_dict(self) -> dict:
        """Convert to the dict format expected by the training loop."""
        return {
            "traj_a": self.image_a.traj_id,
            "step_a": self.image_a.step_key,
            "sigma_a": self.image_a.sigma,
            "traj_b": self.image_b.traj_id,
            "step_b": self.image_b.step_key,
            "sigma_b": self.image_b.sigma,
            "width_a": self.image_a.width,
            "height_a": self.image_a.height,
            "width_b": self.image_b.width,
            "height_b": self.image_b.height,
            "cross_resolution": self.cross_resolution,
            "flops_cost": self.flops_cost,
        }


# ---------------------------------------------------------------------------
# FLOPS allocation across tiers
# ---------------------------------------------------------------------------

def _compute_tier_pair_allocation(
    tier_traj_counts: dict[int, int],
    n_pairs: int,
    tier_flops_targets: dict[int, float] | None = None,
    mega_fraction: float = 0.33,
) -> dict[int, int]:
    """Compute how many pairs to allocate to each populated tier.

    Uses FLOPS-aware allocation: tiers with expensive images get fewer
    pair slots but each pair contributes more FLOPS. Tiers with cheap
    images get more pair slots.

    The allocation minimizes FLOPS allocation error relative to a target.

    Algorithm:
      1. For each tier, compute attention FLOPS ratio:
         ratio = (anchor_pixels / ref_pixels)  -- approximate tokens^2
         More precisely: tokens = anchor / (8*2)^2 = anchor/256.
         ref_tokens = 1024^2 / 256 = 4096. ratio = (tokens/ref_tokens)^2.
      2. Given tier_flops_targets (default: {1048576: 0.33}, rest shares 0.67),
         solve for pair counts that approximate the target FLOPS allocation.
      3. Round to integers, ensuring at least 1 pair from each populated
         tier that has a nonzero FLOPS target.

    Args:
        tier_traj_counts: Dict mapping anchor -> trajectory count (only populated tiers).
        n_pairs: Total pairs to allocate.
        tier_flops_targets: Target FLOPS fractions for specific tiers.
            Default: {1048576: 0.33} with 0.67 shared across the rest.
        mega_fraction: Legacy parameter. Used as the megapixel fraction when
            tier_flops_targets is None.

    Returns:
        Dict mapping anchor -> pair count. Sum equals n_pairs.
    """
    populated = {a: c for a, c in tier_traj_counts.items() if c > 0}
    if not populated:
        return {}
    if n_pairs <= 0:
        return {a: 0 for a in populated}

    anchors = sorted(populated.keys())

    # If only one tier, give it everything
    if len(anchors) == 1:
        return {anchors[0]: n_pairs}

    # Compute FLOPS cost per pair for each tier
    # A pair is 2 images, each with attention cost proportional to tokens^2
    ref_tokens = 1048576 / 256  # = 4096, reference tokens for 1024^2
    tier_flops_per_pair: dict[int, float] = {}
    for anchor in anchors:
        tokens = anchor / 256
        ratio = (tokens / ref_tokens) ** 2
        tier_flops_per_pair[anchor] = 2.0 * ratio  # 2 images per pair

    # Determine target FLOPS fractions
    if tier_flops_targets is None:
        # Legacy: binary mega/small split
        tier_flops_targets = {1048576: mega_fraction}

    # Assign specified fractions to populated tiers
    specified_total = 0.0
    tier_target_frac: dict[int, float] = {}
    for anchor in anchors:
        if anchor in tier_flops_targets:
            tier_target_frac[anchor] = tier_flops_targets[anchor]
            specified_total += tier_flops_targets[anchor]

    # Distribute remaining fraction to unspecified populated tiers
    remaining = max(0.0, 1.0 - specified_total)
    unspecified = [a for a in anchors if a not in tier_target_frac]
    total_unspec_trajs = sum(populated[a] for a in unspecified)
    for a in unspecified:
        if total_unspec_trajs > 0:
            tier_target_frac[a] = remaining * populated[a] / total_unspec_trajs
        else:
            tier_target_frac[a] = 0.0

    # Normalize fractions
    total_frac = sum(tier_target_frac.values())
    if total_frac > 0:
        for a in tier_target_frac:
            tier_target_frac[a] /= total_frac

    # Compute ideal (continuous) pair counts from target FLOPS fractions.
    # target_flops_for_tier = tier_target_frac[a] * total_flops
    # pairs_for_tier * flops_per_pair = target_flops_for_tier
    # => pairs_for_tier = tier_target_frac[a] * total_flops / flops_per_pair
    # But total_flops = sum(pairs_for_tier * flops_per_pair) -- circular.
    # Instead: the fraction of pairs in tier a is:
    # pairs_a / n_pairs = tier_target_frac[a] / flops_per_pair_a / Z
    # where Z = sum(tier_target_frac[b] / flops_per_pair_b) normalizes.
    raw_pair_shares: dict[int, float] = {}
    for a in anchors:
        fpp = tier_flops_per_pair[a]
        if fpp > 0:
            raw_pair_shares[a] = tier_target_frac.get(a, 0.0) / fpp
        else:
            raw_pair_shares[a] = 0.0

    total_shares = sum(raw_pair_shares.values())
    if total_shares <= 0:
        # Fallback: uniform across populated tiers
        per = n_pairs // len(anchors)
        allocation = {a: per for a in anchors}
        # Distribute remainder
        remainder = n_pairs - sum(allocation.values())
        for i, a in enumerate(sorted(anchors, reverse=True)):
            if i < remainder:
                allocation[a] += 1
        return allocation

    # Continuous pair counts
    continuous: dict[int, float] = {
        a: raw_pair_shares[a] / total_shares * n_pairs for a in anchors
    }

    # Round to integers with minimum 1 for tiers with nonzero target
    allocation: dict[int, int] = {}
    for a in anchors:
        if tier_target_frac.get(a, 0.0) > 0 and populated[a] > 0:
            allocation[a] = max(1, round(continuous[a]))
        else:
            allocation[a] = round(continuous[a])

    # Adjust total to exactly n_pairs
    total_alloc = sum(allocation.values())
    if total_alloc != n_pairs:
        diff = n_pairs - total_alloc
        remainders = {a: continuous[a] - allocation[a] for a in anchors}
        if diff > 0:
            # Need more pairs: add to tiers with largest positive remainder
            for _ in range(abs(diff)):
                best = max(remainders, key=lambda a: remainders[a])
                allocation[best] += 1
                remainders[best] -= 1.0
        else:
            # Need fewer: remove from tiers that can afford to lose a pair.
            # Tiers with nonzero targets should keep at least 1 if possible,
            # but if n_pairs < n_tiers we must relax the minimum.
            for _ in range(abs(diff)):
                # Candidates: tiers with allocation > minimum threshold
                candidates = [
                    a for a in remainders
                    if allocation[a] > (1 if tier_target_frac.get(a, 0) > 0 else 0)
                ]
                if not candidates:
                    # Relax: allow reducing to 0 from any tier > 0
                    candidates = [a for a in remainders if allocation[a] > 0]
                if not candidates:
                    break  # Cannot reduce further
                best = min(candidates, key=lambda a: remainders[a])
                allocation[best] -= 1
                remainders[best] += 1.0

    return allocation


# ---------------------------------------------------------------------------
# BTRMPairSampler
# ---------------------------------------------------------------------------

class BTRMPairSampler:
    """Samples (image_a, image_b) pairs on-the-fly from a V2 dataset.

    The pair space is the FULL combinatorial space of all image positions.
    Any two images can be paired -- there are no prompt, resolution, or
    trajectory matching requirements. The only constraint is that the two
    positions are not identical (same trajectory AND same step).

    The pair space is defined by:
    - A set of trajectory IDs (filtered by deprecation defaults)
    - For each trajectory, a set of step positions (each with a sigma/logSNR value)
    - A logSNR-based sampling weight function that biases toward cleaner images

    Pairs are sampled WITHOUT materializing the full pair table. Preferences
    are NOT computed here -- that is the training loop's responsibility, via
    a separate preference_fn. This decoupling allows the same sampler to work
    with any reward function (scrimblo/scrongle, PINKER/TNT, human labels,
    self-training) without changes to the pair selection algorithm.

    6-tier resolution bucketing:
      Trajectories are classified into 6 tiers (MEGAPIXEL_ANCHORS) via
      assign_budget_tier(). The stratified batch sampler allocates pair
      slots across all populated tiers. Cross-resolution pairs are created
      by sampling image B from a different tier than image A.

    Two-stage sampling procedure (avoids N-element softmax):
      1. Pick trajectory A (uniform or FLOPS-weighted by resolution)
      2. Pick step_a within trajectory A (softmax over ~7 step logits)
      3. Pick trajectory B (potentially same trajectory for intra-traj pairs)
      4. Pick step_b within trajectory B (same logit weighting)
      5. Ensure (traj_a, step_a) != (traj_b, step_b)

    Usage:
        from src_ii.pair_sampler import BTRMPairSampler

        sampler = BTRMPairSampler(
            positions=positions,  # from build_positions_from_v2()
        )
        pair = sampler.sample_pair()
        # pair = {"traj_a": 42, "step_a": "step_14", "sigma_a": 0.034,
        #         "traj_b": 17, "step_b": "step_04", "sigma_b": 0.65}
        # Preferences are computed downstream by the training loop.
    """

    def __init__(
        self,
        positions: list[_ImagePosition],
        allow_inter_trajectory: bool = True,
        allow_intra_trajectory: bool = False,
        rng_seed: int | None = None,
        flops_weights: dict[int, float] | None = None,
        clean_fraction: float | None = None,
        clean_decay_scale: float = 0.5,
    ) -> None:
        """
        Args:
            positions: List of _ImagePosition objects (from build_positions_from_v2).
            allow_inter_trajectory: Allow pairs from different trajectories.
            allow_intra_trajectory: Allow pairs from the same trajectory.
            rng_seed: Optional seed for reproducibility.
            flops_weights: Optional per-trajectory FLOPS sampling weights.
                Keys are trajectory IDs, values are unnormalized weights.
                When provided, trajectory selection is weighted by these values
                instead of uniform. Computed by compute_flops_sampling_weights().
                Trajectories not in this dict get weight 0 (excluded).
            clean_fraction: When set (0.0 to 1.0), enables clean-biased step
                selection. With probability clean_fraction, the "final" (sigma=0)
                position is selected. With probability (1 - clean_fraction), a
                non-final position is selected using exponential decay biased
                toward high logSNR. Default None preserves the existing logit-
                based softmax step selection. Set to 0.8 for the 80/20 spec.
            clean_decay_scale: Controls the steepness of exponential decay for
                non-clean positions. Higher = more peaked toward near-clean.
                Default 0.5 gives ~10x preference per 5 logSNR nats.
        """
        if not positions:
            raise ValueError("positions must be non-empty")
        if not allow_inter_trajectory and not allow_intra_trajectory:
            raise ValueError(
                "At least one of allow_inter_trajectory or allow_intra_trajectory "
                "must be True"
            )

        self._allow_inter = allow_inter_trajectory
        self._allow_intra = allow_intra_trajectory
        self._rng = random.Random(rng_seed)

        # Group positions by trajectory
        self._traj_positions: dict[int, list[_ImagePosition]] = {}
        for pos in positions:
            self._traj_positions.setdefault(pos.traj_id, []).append(pos)

        self._traj_ids = sorted(self._traj_positions.keys())
        self._n_trajectories = len(self._traj_ids)

        # Pre-compute per-trajectory step sampling weights (softmax over logits)
        # Each trajectory has ~7 steps, so softmax is trivial.
        self._traj_step_probs: dict[int, np.ndarray] = {}
        for traj_id, traj_positions in self._traj_positions.items():
            logits = np.array([p.logit for p in traj_positions], dtype=np.float64)
            self._traj_step_probs[traj_id] = _softmax(logits)

        # Total number of image positions
        self._n_positions = len(positions)

        # FLOPS-weighted trajectory selection
        self._use_flops_weights = flops_weights is not None
        if self._use_flops_weights:
            raw_weights = np.array(
                [flops_weights.get(tid, 0.0) for tid in self._traj_ids],
                dtype=np.float64,
            )
            total = raw_weights.sum()
            if total <= 0.0:
                self._use_flops_weights = False
                self._traj_probs = None
                self._traj_cdf = None
            else:
                self._traj_probs = raw_weights / total
                self._traj_cdf = np.cumsum(self._traj_probs)
        else:
            self._traj_probs = None
            self._traj_cdf = None

        # Build 6-tier trajectory grouping
        self._tier_traj_ids: dict[int, list[int]] = {a: [] for a in MEGAPIXEL_ANCHORS}
        for traj_id, pos_list in self._traj_positions.items():
            w, h = pos_list[0].width, pos_list[0].height
            anchor = assign_budget_tier(w, h)
            self._tier_traj_ids[anchor].append(traj_id)

        for a in self._tier_traj_ids:
            self._tier_traj_ids[a].sort()

        # Build per-tier CDFs when FLOPS weights are available
        self._tier_traj_cdf: dict[int, np.ndarray | None] = {a: None for a in MEGAPIXEL_ANCHORS}
        if self._use_flops_weights:
            for anchor, tids in self._tier_traj_ids.items():
                if not tids:
                    continue
                raw = np.array([flops_weights.get(tid, 0.0) for tid in tids], dtype=np.float64)
                total = raw.sum()
                if total > 0:
                    probs = raw / total
                    self._tier_traj_cdf[anchor] = np.cumsum(probs)

        # Backward-compatible binary bucket mapping
        # "megapixel" maps to the 1048576 anchor, "small" maps to all others
        from src_ii.flops_sampling import _MEGAPIXEL_THRESHOLD
        self._bucket_traj_ids: dict[str, list[int]] = {"megapixel": [], "small": []}
        for traj_id, pos_list in self._traj_positions.items():
            w, h = pos_list[0].width, pos_list[0].height
            pixels = w * h
            if pixels >= _MEGAPIXEL_THRESHOLD:
                self._bucket_traj_ids["megapixel"].append(traj_id)
            else:
                self._bucket_traj_ids["small"].append(traj_id)
        for bk in self._bucket_traj_ids:
            self._bucket_traj_ids[bk].sort()

        # Legacy binary bucket CDFs
        self._bucket_traj_cdf: dict[str, np.ndarray | None] = {"megapixel": None, "small": None}
        if self._use_flops_weights:
            for bk, tids in self._bucket_traj_ids.items():
                if not tids:
                    continue
                raw = np.array([flops_weights.get(tid, 0.0) for tid in tids], dtype=np.float64)
                total = raw.sum()
                if total > 0:
                    probs = raw / total
                    self._bucket_traj_cdf[bk] = np.cumsum(probs)

        # Stats for observability
        self._n_sampled = 0
        self._n_retried = 0
        self._n_per_tier_sampled: dict[int, int] = {a: 0 for a in MEGAPIXEL_ANCHORS}
        # Legacy stats
        self._n_megapixel_sampled = 0
        self._n_small_sampled = 0

        # Per-tier resolution info for FLOPS-budget macrobatch
        self._traj_resolution: dict[int, tuple[int, int]] = {}
        for traj_id, pos_list in self._traj_positions.items():
            self._traj_resolution[traj_id] = (pos_list[0].width, pos_list[0].height)

        # Clean-biased step selection
        self._clean_fraction = clean_fraction
        self._clean_decay_scale = clean_decay_scale

        # Pre-compute per-trajectory clean-biased data when enabled
        if clean_fraction is not None:
            # For each trajectory, identify the "final" (sigma=0) position
            # and pre-compute exponential decay logits for non-clean positions.
            self._traj_final_idx: dict[int, int | None] = {}
            self._traj_nonfinal_logits: dict[int, np.ndarray] = {}
            self._traj_nonfinal_probs: dict[int, np.ndarray | None] = {}

            for traj_id, traj_positions in self._traj_positions.items():
                # Find the best "clean" position: prefer sigma=0, else highest logSNR
                best_clean_idx = None
                best_logsnr = float("-inf")
                for i, pos in enumerate(traj_positions):
                    if pos.sigma <= 0.0:
                        best_clean_idx = i
                        break  # sigma=0 is always the best clean candidate
                    logsnr = _sigma_to_logsnr(pos.sigma)
                    if logsnr > best_logsnr:
                        best_logsnr = logsnr
                        best_clean_idx = i
                self._traj_final_idx[traj_id] = best_clean_idx

                # Compute exponential decay logits for non-clean sampling
                logits = _clean_biased_step_logits(
                    traj_positions, decay_scale=clean_decay_scale,
                )
                self._traj_nonfinal_logits[traj_id] = logits

                # Check if there are any valid non-clean positions
                finite_mask = np.isfinite(logits)
                if finite_mask.any():
                    self._traj_nonfinal_probs[traj_id] = _softmax(
                        logits[finite_mask]
                    )
                else:
                    # All positions are sigma=0; non-clean path has nowhere to go
                    self._traj_nonfinal_probs[traj_id] = None

        # LogSNR histogram counters (lightweight, always active)
        self._logsnr_histogram: dict[str, int] = {}
        for label, _, _ in _LOGSNR_HIST_BINS:
            self._logsnr_histogram[label] = 0
        self._logsnr_histogram[_SIGMA_ZERO_BIN] = 0
        self._n_positions_sampled = 0  # total individual positions sampled

    def _pick_trajectory(
        self,
        resolution_bucket: str | None = None,
        tier_anchor: int | None = None,
    ) -> int:
        """Pick a trajectory ID, optionally constrained to a tier or bucket.

        Args:
            resolution_bucket: Legacy binary bucket ("megapixel" or "small").
            tier_anchor: 6-tier anchor pixel count (from MEGAPIXEL_ANCHORS).
                Takes precedence over resolution_bucket if both are specified.

        Returns:
            A trajectory ID.
        """
        # 6-tier mode
        if tier_anchor is not None:
            tids = self._tier_traj_ids.get(tier_anchor, [])
            if not tids:
                raise RuntimeError(
                    f"BTRMPairSampler: no trajectories in tier {tier_anchor}. "
                    f"Populated tiers: {[a for a, t in self._tier_traj_ids.items() if t]}"
                )
            cdf = self._tier_traj_cdf.get(tier_anchor)
            if cdf is not None:
                idx = np.searchsorted(cdf, self._rng.random())
                idx = min(idx, len(tids) - 1)
                return tids[idx]
            else:
                return self._rng.choice(tids)

        # Legacy binary bucket mode
        if resolution_bucket is not None:
            tids = self._bucket_traj_ids.get(resolution_bucket, [])
            if not tids:
                raise RuntimeError(
                    f"BTRMPairSampler: no trajectories in bucket '{resolution_bucket}'. "
                    f"Available buckets: megapixel={len(self._bucket_traj_ids['megapixel'])}, "
                    f"small={len(self._bucket_traj_ids['small'])}"
                )
            cdf = self._bucket_traj_cdf.get(resolution_bucket)
            if cdf is not None:
                idx = np.searchsorted(cdf, self._rng.random())
                idx = min(idx, len(tids) - 1)
                return tids[idx]
            else:
                return self._rng.choice(tids)

        # Global FLOPS-weighted or uniform
        if self._use_flops_weights:
            idx = np.searchsorted(self._traj_cdf, self._rng.random())
            idx = min(idx, len(self._traj_ids) - 1)
            return self._traj_ids[idx]
        else:
            return self._rng.choice(self._traj_ids)

    def _pick_position(
        self,
        resolution_bucket: str | None = None,
        tier_anchor: int | None = None,
        use_clean_bias: bool | None = None,
    ) -> _ImagePosition:
        """Two-stage sampling: pick trajectory, then pick step weighted.

        Args:
            resolution_bucket: Legacy binary bucket constraint.
            tier_anchor: 6-tier anchor constraint.
            use_clean_bias: If True, use clean-biased step selection for this
                position. If None, uses the instance's clean_fraction setting.
                If False, always uses the legacy logit-softmax path.
        """
        traj_id = self._pick_trajectory(
            resolution_bucket=resolution_bucket, tier_anchor=tier_anchor,
        )

        # Determine whether to use clean-biased sampling
        _use_clean = (
            use_clean_bias is True
            or (use_clean_bias is None and self._clean_fraction is not None)
        )

        if _use_clean:
            pos = self.sample_step_with_clean_bias(traj_id)
        else:
            positions = self._traj_positions[traj_id]
            probs = self._traj_step_probs[traj_id]
            idx = np.searchsorted(np.cumsum(probs), self._rng.random())
            idx = min(idx, len(positions) - 1)
            pos = positions[idx]

        # Track logSNR histogram
        self._record_position_sample(pos)
        return pos

    def _record_position_sample(self, pos: _ImagePosition) -> None:
        """Record one sampled position in the logSNR histogram."""
        self._n_positions_sampled += 1
        bin_label = _logsnr_hist_bin(pos.sigma)
        self._logsnr_histogram[bin_label] = self._logsnr_histogram.get(bin_label, 0) + 1

    def sample_step_with_clean_bias(
        self,
        traj_id: int,
        clean_fraction: float | None = None,
    ) -> _ImagePosition:
        """Sample a step from a trajectory using clean-biased 80/20 logic.

        With probability clean_fraction:
          - Select the "final" (sigma=0) position if available
          - Fall back to the highest-logSNR position if no sigma=0 exists

        With probability (1 - clean_fraction):
          - Select from non-final positions using exponential decay biased
            toward high logSNR (slightly noisy). Steeper decay = stronger
            bias toward near-clean images.

        Args:
            traj_id: Trajectory ID to sample from.
            clean_fraction: Override the instance's clean_fraction for this
                call. If None, uses self._clean_fraction. Must be set if
                self._clean_fraction is None (i.e., clean-biased mode was
                not enabled at init time).

        Returns:
            The sampled _ImagePosition.

        Raises:
            ValueError: If clean_fraction is None and the instance was not
                initialized with clean_fraction.
        """
        cf = clean_fraction if clean_fraction is not None else self._clean_fraction
        if cf is None:
            raise ValueError(
                "sample_step_with_clean_bias() called but clean_fraction is None. "
                "Either pass clean_fraction= or initialize BTRMPairSampler with "
                "clean_fraction=0.8."
            )

        positions = self._traj_positions[traj_id]

        # If the trajectory was pre-indexed at init (clean_fraction was set at init)
        if hasattr(self, '_traj_final_idx') and traj_id in self._traj_final_idx:
            final_idx = self._traj_final_idx[traj_id]
            nonfinal_logits = self._traj_nonfinal_logits[traj_id]
            nonfinal_probs = self._traj_nonfinal_probs[traj_id]
        else:
            # Compute on-the-fly (clean_fraction passed at call time, not init)
            final_idx = None
            best_logsnr = float("-inf")
            for i, pos in enumerate(positions):
                if pos.sigma <= 0.0:
                    final_idx = i
                    break
                logsnr = _sigma_to_logsnr(pos.sigma)
                if logsnr > best_logsnr:
                    best_logsnr = logsnr
                    final_idx = i

            nonfinal_logits = _clean_biased_step_logits(
                positions, decay_scale=self._clean_decay_scale,
            )
            finite_mask = np.isfinite(nonfinal_logits)
            if finite_mask.any():
                nonfinal_probs = _softmax(nonfinal_logits[finite_mask])
            else:
                nonfinal_probs = None

        # Coin flip: clean or non-clean?
        if self._rng.random() < cf:
            # Clean path: select the final/cleanest position
            if final_idx is not None:
                return positions[final_idx]
            # Fallback: should not happen (final_idx is always set above)
            return positions[0]
        else:
            # Non-clean path: exponential decay toward high logSNR
            if nonfinal_probs is None:
                # All positions are sigma=0: fall back to the final position
                if final_idx is not None:
                    return positions[final_idx]
                return positions[0]

            # Map from compressed (non-clean-only) index to full position index
            finite_mask = np.isfinite(nonfinal_logits)
            nonfinal_indices = np.where(finite_mask)[0]

            cdf = np.cumsum(nonfinal_probs)
            r = self._rng.random()
            compressed_idx = int(np.searchsorted(cdf, r))
            compressed_idx = min(compressed_idx, len(nonfinal_indices) - 1)
            full_idx = nonfinal_indices[compressed_idx]
            return positions[full_idx]

    def get_logsnr_histogram(self) -> dict[str, int]:
        """Return the logSNR distribution histogram of sampled positions.

        Returns a dict mapping bin labels to counts. Bins:
          [-inf,-5), [-5,-2), [-2,0), [0,2), [2,5), [5,8), [8,+inf), sigma=0
        """
        return dict(self._logsnr_histogram)

    def get_clean_fraction(self) -> float:
        """Return the actual measured fraction of sigma=0 positions sampled.

        Returns 0.0 if no positions have been sampled yet.
        """
        if self._n_positions_sampled == 0:
            return 0.0
        return self._logsnr_histogram.get(_SIGMA_ZERO_BIN, 0) / self._n_positions_sampled

    def sample_pair(
        self,
        resolution_bucket: str | None = None,
        tier_anchor: int | None = None,
    ) -> dict:
        """Sample one (image_a, image_b) pair.

        Args:
            resolution_bucket: Legacy binary bucket constraint.
            tier_anchor: 6-tier anchor constraint (takes precedence).

        Returns dict with keys:
          - traj_a, traj_b: trajectory IDs
          - step_a, step_b: step keys (e.g. "step_14", "final")
          - sigma_a, sigma_b: sigma values for the sampled positions
        """
        max_retries = 50

        for attempt in range(max_retries):
            pos_a = self._pick_position(
                resolution_bucket=resolution_bucket, tier_anchor=tier_anchor,
            )

            if not self._allow_inter and self._allow_intra:
                # Intra-trajectory only: sample pos_b from same trajectory
                if self._clean_fraction is not None:
                    pos_b = self.sample_step_with_clean_bias(pos_a.traj_id)
                    self._record_position_sample(pos_b)
                else:
                    positions_b = self._traj_positions[pos_a.traj_id]
                    probs_b = self._traj_step_probs[pos_a.traj_id]
                    idx = np.searchsorted(np.cumsum(probs_b), self._rng.random())
                    idx = min(idx, len(positions_b) - 1)
                    pos_b = positions_b[idx]
                    self._record_position_sample(pos_b)
            elif self._allow_inter and not self._allow_intra:
                while True:
                    pos_b = self._pick_position(
                        resolution_bucket=resolution_bucket, tier_anchor=tier_anchor,
                    )
                    if pos_b.traj_id != pos_a.traj_id:
                        break
            else:
                pos_b = self._pick_position(
                    resolution_bucket=resolution_bucket, tier_anchor=tier_anchor,
                )

            if pos_a.traj_id == pos_b.traj_id and pos_a.step_key == pos_b.step_key:
                self._n_retried += 1
                continue

            pair = {
                "traj_a": pos_a.traj_id,
                "step_a": pos_a.step_key,
                "sigma_a": pos_a.sigma,
                "traj_b": pos_b.traj_id,
                "step_b": pos_b.step_key,
                "sigma_b": pos_b.sigma,
            }

            self._n_sampled += 1
            # Track per-tier stats
            anchor_a = assign_budget_tier(pos_a.width, pos_a.height)
            self._n_per_tier_sampled[anchor_a] = self._n_per_tier_sampled.get(anchor_a, 0) + 1
            # Legacy stats
            if resolution_bucket == "megapixel" or (tier_anchor is not None and tier_anchor >= 1048576):
                self._n_megapixel_sampled += 1
            elif resolution_bucket == "small" or (tier_anchor is not None and tier_anchor < 1048576):
                self._n_small_sampled += 1
            return pair

        raise RuntimeError(
            f"BTRMPairSampler: failed to sample a valid pair after {max_retries} "
            f"retries (resolution_bucket={resolution_bucket!r}, tier_anchor={tier_anchor!r}). "
            f"This suggests the position set is too small or constraints "
            f"are too restrictive."
        )

    def sample_batch(self, n: int) -> list[dict]:
        """Sample n pairs."""
        return [self.sample_pair() for _ in range(n)]

    def sample_stratified_batch(
        self,
        n_pairs: int,
        mega_fraction: float = 0.33,
        tier_flops_targets: dict[int, float] | None = None,
    ) -> list[dict]:
        """Sample a stratified macrobatch enforcing resolution tier allocation.

        Allocates pair slots across all populated resolution tiers to
        approximate a target FLOPS distribution. With 6 tiers instead of
        binary mega/small, the allocation can balance FLOPS across the
        full resolution spectrum.

        The allocation is computed by _compute_tier_pair_allocation(),
        which accounts for the quadratic attention cost of each tier.

        Within each tier, pairs are sampled using the tier-specific CDF
        (FLOPS-weighted within the tier).

        Graceful degradation:
          - If only one tier is populated, all pairs come from that tier.
          - If n_pairs=1, it goes to the tier with the highest FLOPS target.

        Args:
            n_pairs: Total number of pairs to sample.
            mega_fraction: Target FLOPS fraction for the megapixel tier.
                Default 0.33. Used as shorthand for {1048576: 0.33}.
            tier_flops_targets: Full 6-tier FLOPS targets. If provided,
                overrides mega_fraction. Dict mapping anchor -> target fraction.

        Returns:
            Shuffled list of n_pairs pair dicts.
        """
        # Determine tier trajectory counts (only populated tiers)
        tier_counts = {
            a: len(tids) for a, tids in self._tier_traj_ids.items() if tids
        }

        if not tier_counts:
            raise RuntimeError(
                "BTRMPairSampler.sample_stratified_batch: no trajectories "
                "in any resolution tier."
            )

        # If only one tier, give it everything
        if len(tier_counts) == 1:
            anchor = next(iter(tier_counts))
            return [self.sample_pair(tier_anchor=anchor) for _ in range(n_pairs)]

        # Compute allocation
        if tier_flops_targets is None:
            tier_flops_targets = {1048576: mega_fraction}

        allocation = _compute_tier_pair_allocation(
            tier_counts, n_pairs,
            tier_flops_targets=tier_flops_targets,
        )

        # Sample from each tier
        pairs = []
        for anchor, count in allocation.items():
            for _ in range(count):
                pairs.append(self.sample_pair(tier_anchor=anchor))

        # Shuffle so tier pairs are distributed randomly across microbatches
        self._rng.shuffle(pairs)
        return pairs

    # -------------------------------------------------------------------
    # FLOPS-budget macrobatch sampling
    # -------------------------------------------------------------------

    def _sample_pair_spec(
        self,
        tier_anchor: int | None = None,
        allow_cross_resolution: bool = False,
    ) -> PairSpec:
        """Sample a pair and return it as a PairSpec with FLOPS metadata.

        When allow_cross_resolution=True and the dataset has multiple
        populated resolution tiers, there is a 30% chance of sampling
        a cross-resolution pair: image A from one tier, image B from a
        DIFFERENT tier. No prompt matching is required -- pinkify and
        thisnotthat are universal scoring functions that apply to any
        image regardless of content, prompt, or resolution.

        Args:
            tier_anchor: Constrain image A to this tier. Image B may come
                from a different tier if cross-resolution is enabled.
            allow_cross_resolution: If True, allow image B from a different
                tier than image A.

        Returns:
            PairSpec with both positions, FLOPS cost, and cross-res flag.
        """
        max_retries = 50
        populated = self.populated_tiers

        for attempt in range(max_retries):
            pos_a = self._pick_position(tier_anchor=tier_anchor)

            # Decide whether to attempt cross-resolution
            cross_res_attempted = False
            if (
                allow_cross_resolution
                and len(populated) > 1
                and self._rng.random() < 0.30
            ):
                # Pick image B from a DIFFERENT tier than image A.
                # No prompt matching -- any two images can be compared.
                anchor_a = assign_budget_tier(pos_a.width, pos_a.height)
                other_tiers = [a for a in populated if a != anchor_a]
                if other_tiers:
                    cross_res_attempted = True
                    other_tier = self._rng.choice(other_tiers)
                    pos_b = self._pick_position(tier_anchor=other_tier)

            if not cross_res_attempted:
                # Same-tier pair (or unconstrained if tier_anchor is None)
                if not self._allow_inter and self._allow_intra:
                    if self._clean_fraction is not None:
                        pos_b = self.sample_step_with_clean_bias(pos_a.traj_id)
                        self._record_position_sample(pos_b)
                    else:
                        positions_b = self._traj_positions[pos_a.traj_id]
                        probs_b = self._traj_step_probs[pos_a.traj_id]
                        idx = np.searchsorted(np.cumsum(probs_b), self._rng.random())
                        idx = min(idx, len(positions_b) - 1)
                        pos_b = positions_b[idx]
                        self._record_position_sample(pos_b)
                elif self._allow_inter and not self._allow_intra:
                    while True:
                        pos_b = self._pick_position(tier_anchor=tier_anchor)
                        if pos_b.traj_id != pos_a.traj_id:
                            break
                else:
                    pos_b = self._pick_position(tier_anchor=tier_anchor)

            # Reject identical positions
            if pos_a.traj_id == pos_b.traj_id and pos_a.step_key == pos_b.step_key:
                self._n_retried += 1
                continue

            is_cross = (
                assign_budget_tier(pos_a.width, pos_a.height)
                != assign_budget_tier(pos_b.width, pos_b.height)
            )
            cost = pair_flops_units(
                pos_a.width, pos_a.height,
                pos_b.width, pos_b.height,
            )

            self._n_sampled += 1
            anchor_a = assign_budget_tier(pos_a.width, pos_a.height)
            self._n_per_tier_sampled[anchor_a] = self._n_per_tier_sampled.get(anchor_a, 0) + 1

            return PairSpec(
                image_a=pos_a,
                image_b=pos_b,
                flops_cost=cost,
                cross_resolution=is_cross,
            )

        raise RuntimeError(
            f"BTRMPairSampler._sample_pair_spec: failed after {max_retries} retries. "
            f"tier_anchor={tier_anchor}, allow_cross_resolution={allow_cross_resolution}"
        )

    def _top_populated_tier(self) -> int:
        """Return the anchor of the most expensive populated tier."""
        for anchor in reversed(MEGAPIXEL_ANCHORS):
            if self._tier_traj_ids.get(anchor):
                return anchor
        raise RuntimeError("No populated tiers")

    def sample_macrobatch(
        self,
        budget_units: float = 3.0,
        tier_flops_targets: dict[int, float] | None = None,
        allow_cross_resolution: bool = True,
        top_tier_guarantee: bool = True,
    ) -> list[PairSpec]:
        """Sample a FLOPS-budget macrobatch.

        A macrobatch is defined by a FLOPS budget in 1024^2-equivalent units,
        NOT by a fixed pair count. The number of pairs varies per call
        depending on the resolution mix.

        Algorithm:
          1. If top_tier_guarantee, sample one pair from the top populated
             tier. This ensures every macrobatch has at least one
             high-resolution pair for rich gradient signal.
          2. Fill remaining budget by sampling tiers weighted by how much
             of their FLOPS target remains unfilled. This prevents any
             tier from being aggressively undersampled.
          3. Shuffle the result so bin packing is not biased by sampling
             order.

        FLOPS unit definition:
          1.0 unit = attention FLOPS of one 1024x1024 image.
          A pair of 1024x1024 images costs ~2.0 units.
          A pair of 256x256 images costs ~0.008 units.

        Args:
            budget_units: Target FLOPS budget in 1024^2-equiv units.
                Default 3.0 (roughly 1.5 megapixel pairs or ~400 small pairs).
            tier_flops_targets: Target FLOPS fractions per tier.
                Default: {1048576: 0.33} with rest distributed proportionally.
            allow_cross_resolution: If True, allow cross-resolution pairs
                (images from different resolution tiers).
            top_tier_guarantee: If True, guarantee at least one pair from
                the top populated tier. Default True.

        Returns:
            Shuffled list of PairSpec objects. Length varies per call.
        """
        if tier_flops_targets is None:
            tier_flops_targets = {1048576: 0.33}

        populated = self.populated_tiers
        if not populated:
            raise RuntimeError("No populated tiers for macrobatch sampling")

        pairs: list[PairSpec] = []
        consumed = 0.0

        # Track per-tier FLOPS consumed for balanced sampling
        tier_consumed: dict[int, float] = {a: 0.0 for a in populated}

        # Phase 1: Guarantee one pair from the top populated tier
        if top_tier_guarantee:
            top_tier = self._top_populated_tier()
            pair = self._sample_pair_spec(
                tier_anchor=top_tier,
                allow_cross_resolution=allow_cross_resolution,
            )
            pairs.append(pair)
            consumed += pair.flops_cost
            tier_consumed[top_tier] = tier_consumed.get(top_tier, 0.0) + pair.flops_cost

        # Compute target FLOPS fractions for all populated tiers
        specified_total = 0.0
        tier_target_frac: dict[int, float] = {}
        for anchor in populated:
            if anchor in tier_flops_targets:
                tier_target_frac[anchor] = tier_flops_targets[anchor]
                specified_total += tier_flops_targets[anchor]

        remaining_frac = max(0.0, 1.0 - specified_total)
        unspecified = [a for a in populated if a not in tier_target_frac]
        total_unspec_trajs = sum(
            len(self._tier_traj_ids[a]) for a in unspecified
        )
        for a in unspecified:
            if total_unspec_trajs > 0:
                tier_target_frac[a] = remaining_frac * len(self._tier_traj_ids[a]) / total_unspec_trajs
            else:
                tier_target_frac[a] = 0.0

        # Normalize
        total_frac = sum(tier_target_frac.values())
        if total_frac > 0:
            for a in tier_target_frac:
                tier_target_frac[a] /= total_frac

        # Phase 2: Fill remaining budget
        # Safety: cap at 1000 pairs to prevent runaway on degenerate inputs
        max_pairs = 1000
        while consumed < budget_units and len(pairs) < max_pairs:
            # Weight each tier by: target_flops - consumed_flops (clamped to 0)
            # This ensures tiers that are behind their target get preferentially
            # sampled. If all tiers are above target, fall back to proportional.
            tier_weights: dict[int, float] = {}
            for a in populated:
                target_flops_abs = tier_target_frac.get(a, 0.0) * budget_units
                deficit = max(0.0, target_flops_abs - tier_consumed.get(a, 0.0))
                # Add a small floor to prevent any tier from being totally excluded
                tier_weights[a] = deficit + 0.001

            # Weighted tier selection
            weight_list = [tier_weights[a] for a in populated]
            total_w = sum(weight_list)
            probs = [w / total_w for w in weight_list]
            r = self._rng.random()
            cumsum = 0.0
            selected_tier = populated[-1]  # fallback
            for i, a in enumerate(populated):
                cumsum += probs[i]
                if r <= cumsum:
                    selected_tier = a
                    break

            pair = self._sample_pair_spec(
                tier_anchor=selected_tier,
                allow_cross_resolution=allow_cross_resolution,
            )
            pairs.append(pair)
            consumed += pair.flops_cost
            tier_consumed[selected_tier] = tier_consumed.get(selected_tier, 0.0) + pair.flops_cost

        # Phase 3: Shuffle so packing isn't biased by sampling order
        self._rng.shuffle(pairs)

        return pairs

    @property
    def n_trajectories(self) -> int:
        return self._n_trajectories

    @property
    def n_positions(self) -> int:
        return self._n_positions

    @property
    def pair_space_size(self) -> int:
        """Approximate number of distinct pairs (N*(N-1)/2)."""
        n = self._n_positions
        return n * (n - 1) // 2

    @property
    def has_megapixel(self) -> bool:
        """Whether the dataset contains any megapixel trajectories."""
        return len(self._bucket_traj_ids.get("megapixel", [])) > 0

    @property
    def has_small(self) -> bool:
        """Whether the dataset contains any small-resolution trajectories."""
        return len(self._bucket_traj_ids.get("small", [])) > 0

    @property
    def populated_tiers(self) -> list[int]:
        """List of anchor pixel counts that have trajectories."""
        return sorted(a for a, tids in self._tier_traj_ids.items() if tids)

    @property
    def tier_trajectory_counts(self) -> dict[int, int]:
        """Dict mapping anchor -> trajectory count for populated tiers."""
        return {a: len(tids) for a, tids in self._tier_traj_ids.items() if tids}

    def stats(self) -> dict:
        """Return sampling statistics for observability."""
        result = {
            "n_trajectories": self._n_trajectories,
            "n_positions": self._n_positions,
            "pair_space_size": self.pair_space_size,
            "n_sampled": self._n_sampled,
            "n_retried": self._n_retried,
            "retry_rate": self._n_retried / max(self._n_sampled + self._n_retried, 1),
            "n_megapixel_sampled": self._n_megapixel_sampled,
            "n_small_sampled": self._n_small_sampled,
            "n_megapixel_trajectories": len(self._bucket_traj_ids.get("megapixel", [])),
            "n_small_trajectories": len(self._bucket_traj_ids.get("small", [])),
            "populated_tiers": self.populated_tiers,
            "tier_trajectory_counts": self.tier_trajectory_counts,
            "tier_sample_counts": dict(self._n_per_tier_sampled),
        }
        # Include logSNR histogram and clean fraction if positions were sampled
        if self._n_positions_sampled > 0:
            result["logsnr_histogram"] = self.get_logsnr_histogram()
            result["measured_clean_fraction"] = self.get_clean_fraction()
            result["n_positions_sampled"] = self._n_positions_sampled
        if self._clean_fraction is not None:
            result["clean_fraction_setting"] = self._clean_fraction
        return result


# ---------------------------------------------------------------------------
# Position builders: construct _ImagePosition lists from V2 datasets
# ---------------------------------------------------------------------------

def build_positions_from_v2(
    dataset_reader: Any,  # DatasetReader instance
    traj_ids: list[int] | None = None,
    threshold: float = 10.0,
    interval: float = 5.0,
    decay_rate: float = 0.5,
) -> list[_ImagePosition]:
    """Build image positions from a V2 dataset reader.

    For each trajectory and each step within it, computes the sigma value
    from the sigma schedule and derives a logSNR sampling logit using
    the geometric decay function.

    Args:
        dataset_reader: A DatasetReader instance (from futudiffu.dataset_v2).
        traj_ids: Subset of trajectory IDs to include. If None, uses all
            trajectories in the reader.
        threshold: logSNR value where decay begins (default 10.0).
        interval: logSNR nats per decay step (default 5.0).
        decay_rate: multiplicative factor per interval (default 0.5).

    Returns:
        List of _ImagePosition objects.
    """
    import torch
    from src_ii.sigma_schedule import build_sigma_schedule, resolution_shift

    positions = []

    if traj_ids is None:
        traj_ids = dataset_reader._table.column("traj_id").to_pylist()

    for traj_id in traj_ids:
        meta, accessor = dataset_reader[traj_id]
        n_steps = meta["n_steps"]

        # Extract resolution from trajectory metadata
        traj_w = meta.get("width", 1280)
        traj_h = meta.get("height", 832)
        traj_prompt_idx = meta.get("prompt_idx", -1)

        # Build sigma schedule for this trajectory with resolution-dependent shift.
        denoise = meta.get("denoise") or 1.0
        recorded_shift = meta.get("sampling_shift")
        if recorded_shift is not None:
            shift = float(recorded_shift)
        else:
            shift = resolution_shift(traj_w, traj_h)
        sigmas = build_sigma_schedule(
            n_steps, sampling_shift=shift, denoise=denoise,
            device="cpu", dtype=torch.float32,
        )

        for step_label in accessor.available_steps:
            if step_label == "final":
                # The final denoised image has sigma=0.0 (terminal).
                # sigmas[-1] = 0.0 is the terminal value; sigmas[-2] is the
                # last step's INPUT sigma, NOT the final image's sigma.
                sigma_val = 0.0
            else:
                step_idx = int(step_label.split("_")[1])
                if step_idx < len(sigmas):
                    sigma_val = float(sigmas[step_idx].item())
                else:
                    sigma_val = 0.01

            logit = logsnr_sampling_logit(
                sigma_val,
                threshold=threshold,
                interval=interval,
                decay_rate=decay_rate,
            )
            positions.append(_ImagePosition(
                traj_id, step_label, sigma_val, logit,
                width=traj_w, height=traj_h,
                prompt_idx=traj_prompt_idx,
            ))

    return positions


def build_positions_from_manifest(
    manifest_records: list[dict],
    trajectory_indices: list[int],
    dataset_base_dir: Any,  # Path
    threshold: float = 10.0,
    interval: float = 5.0,
    decay_rate: float = 0.5,
) -> list[_ImagePosition]:
    """Build image positions from a V1-style manifest (backward compat).

    Scans the trajectory directories for step_*.pt and final.pt files
    to determine available steps.

    Args:
        manifest_records: List of trajectory records from manifest.json.
        trajectory_indices: Which trajectory indices to include.
        dataset_base_dir: Path to the btrm_dataset/ directory.
        threshold: logSNR value where decay begins (default 10.0).
        interval: logSNR nats per decay step (default 5.0).
        decay_rate: multiplicative factor per interval (default 0.5).

    Returns:
        List of _ImagePosition objects.
    """
    import torch
    from pathlib import Path
    from src_ii.sigma_schedule import build_sigma_schedule

    base = Path(dataset_base_dir)
    positions = []

    for traj_idx in trajectory_indices:
        if traj_idx >= len(manifest_records):
            continue
        record = manifest_records[traj_idx]
        n_steps = record["n_steps"]

        traj_w = record.get("width", 1280)
        traj_h = record.get("height", 832)
        traj_prompt_idx = record.get("prompt_idx", -1)

        traj_dir = base / "latents" / f"traj_{traj_idx:06d}"
        if not traj_dir.exists():
            continue

        sigmas = build_sigma_schedule(n_steps, device="cpu", dtype=torch.float32)

        step_files = sorted(traj_dir.glob("step_*.pt"))
        for sf in step_files:
            step_key = sf.stem
            step_idx = int(step_key.split("_")[1])
            if step_idx < len(sigmas):
                sigma_val = float(sigmas[step_idx].item())
            else:
                sigma_val = 0.01
            logit = logsnr_sampling_logit(
                sigma_val,
                threshold=threshold,
                interval=interval,
                decay_rate=decay_rate,
            )
            positions.append(_ImagePosition(
                traj_idx, step_key, sigma_val, logit,
                width=traj_w, height=traj_h,
                prompt_idx=traj_prompt_idx,
            ))

        final_file = traj_dir / "final.pt"
        if final_file.exists():
            sigma_val = float(sigmas[-2].item()) if len(sigmas) > 1 else 0.01
            logit = logsnr_sampling_logit(
                sigma_val,
                threshold=threshold,
                interval=interval,
                decay_rate=decay_rate,
            )
            positions.append(_ImagePosition(
                traj_idx, "final", sigma_val, logit,
                width=traj_w, height=traj_h,
                prompt_idx=traj_prompt_idx,
            ))

    return positions

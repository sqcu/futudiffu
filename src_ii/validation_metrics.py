"""Multi-indexed validation covariance tracker for BTRM training.

Tracks per-pair training results indexed by resolution bucket, sigma/logSNR
bucket, head name, aspect ratio bucket, and trajectory source. Maintains
running statistics (mean, variance, count) via Welford's online algorithm,
avoiding unbounded memory growth even for 10K+ steps.

The tracker answers the question: "does funfetti-sampled training have better
accuracy-per-NFE than monotonic megapixel sampling, broken down by resolution
bucket?"

Design choices:
  - Welford's algorithm for numerically stable online mean/variance.
  - Multi-key indexing via string-keyed flat dict of accumulators. Each
    unique combination of (resolution_bucket, logsnr_bucket, head, source)
    gets its own accumulator. This avoids a deeply nested dict structure
    and makes serialization trivial.
  - Resolution is bucketed by megapixels, not by exact (W, H). This
    collapses aspect ratio variants into the same bucket (e.g., 1280x832
    and 832x1280 are both ~1.0 MP).
  - LogSNR bucketing uses the same formula as pair_sampler:
    logSNR = -2 * log(sigma) for sigma > 0, +inf for sigma = 0.
  - Aspect ratio is tracked as a continuous float (W/H) but bucketed
    into coarse bins for cross-tabulation.
  - Trajectory source is an opaque string label from dataset metadata
    (e.g., "original_v1", "gpu_rollout", "policy_rollout").

Import constraints:
  - math, json for pure-Python operations
  - No torch, no numpy (this is a metrics tracker, not a compute module)
  - No futudiffu imports (standalone)
"""

from __future__ import annotations

import json
import math
from typing import Any


# ---------------------------------------------------------------------------
# Bucketing functions
# ---------------------------------------------------------------------------

# Resolution buckets by megapixels.
# Each bucket is (label, lower_bound_mp, upper_bound_mp).
# The bounds are exclusive lower / inclusive upper, except the first
# (inclusive lower) and last (unbounded upper).
_RESOLUTION_BUCKETS: list[tuple[str, float, float]] = [
    ("< 0.1 MP",   0.0,  0.1),   # 256x256 = 0.065 MP
    ("0.1-0.2 MP", 0.1,  0.2),   # 320x320 = 0.102 MP
    ("0.2-0.4 MP", 0.2,  0.4),   # 384x384 = 0.147 MP, 512x512 = 0.262 MP
    ("0.4-0.8 MP", 0.4,  0.8),   # 704x704 = 0.496 MP
    ("0.8-1.2 MP", 0.8,  1.2),   # 1024x1024 = 1.049 MP, 1280x832 = 1.064 MP
    (">= 1.2 MP",  1.2,  float("inf")),
]

# LogSNR buckets. logSNR = -2 * log(sigma) for sigma in (0, 1).
# sigma = 0 maps to logSNR = +inf (special "clean" bucket).
_LOGSNR_BUCKETS: list[tuple[str, float, float]] = [
    ("very_noisy (logSNR < -2)",      float("-inf"), -2.0),
    ("noisy (-2 <= logSNR < 0)",      -2.0,           0.0),
    ("moderate (0 <= logSNR < 2)",     0.0,           2.0),
    ("clean-ish (2 <= logSNR < 5)",    2.0,           5.0),
    ("near-clean (5 <= logSNR < inf)", 5.0,         float("inf")),
    ("clean (sigma=0)",                float("inf"), float("inf")),  # sentinel
]

# Aspect ratio buckets (W/H).
_ASPECT_BUCKETS: list[tuple[str, float, float]] = [
    ("portrait (< 0.8)",   0.0,  0.8),
    ("square (0.8-1.2)",   0.8,  1.2),
    ("landscape (>= 1.2)", 1.2,  float("inf")),
]


def resolution_bucket(width: int, height: int) -> str:
    """Quantize (width, height) into a megapixel bucket label.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Bucket label string.
    """
    mp = (width * height) / 1_000_000.0
    for label, lo, hi in _RESOLUTION_BUCKETS:
        if lo <= mp < hi:
            return label
    return _RESOLUTION_BUCKETS[-1][0]  # fallback to largest


def logsnr_bucket(sigma: float) -> str:
    """Quantize a sigma value into a logSNR bucket label.

    Uses the CONST noise model formula: logSNR = 2 * ln((1-sigma)/sigma).
    This is consistent with pair_sampler.logsnr_sampling_logit() which uses
    the same formula for step weighting. The previous formula (-2*ln(sigma))
    was an incorrect approximation that diverges for sigma > 0.3.

    sigma = 0 maps to the special "clean" bucket.
    sigma >= 1 maps to the "very noisy" bucket.

    Args:
        sigma: Noise level from the CONST diffusion schedule (0 <= sigma <= 1).

    Returns:
        Bucket label string.
    """
    if sigma <= 0.0:
        return "clean (sigma=0)"
    if sigma >= 1.0:
        return "very_noisy (logSNR < -2)"

    logsnr = 2.0 * math.log((1.0 - sigma) / sigma)

    for label, lo, hi in _LOGSNR_BUCKETS:
        if label == "clean (sigma=0)":
            continue  # skip sentinel
        if lo <= logsnr < hi:
            return label
    # Fallback: if logsnr is exactly at a boundary, use the last non-sentinel
    return _LOGSNR_BUCKETS[-2][0]


def aspect_bucket(width: int, height: int) -> str:
    """Quantize aspect ratio (W/H) into a coarse bucket.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Bucket label string.
    """
    if height <= 0:
        return "square (0.8-1.2)"
    ratio = width / height
    for label, lo, hi in _ASPECT_BUCKETS:
        if lo <= ratio < hi:
            return label
    return _ASPECT_BUCKETS[-1][0]


# ---------------------------------------------------------------------------
# Welford accumulator
# ---------------------------------------------------------------------------

class WelfordAccumulator:
    """Online mean/variance using Welford's algorithm.

    Numerically stable. O(1) memory per accumulator regardless of
    sample count. Computes population variance (not sample variance).

    Reference: Welford (1962), "Note on a Method for Calculating
    Corrected Sums of Squares and Products."
    """
    __slots__ = ("count", "mean", "m2")

    def __init__(self, count: int = 0, mean: float = 0.0, m2: float = 0.0):
        self.count = count
        self.mean = mean
        self.m2 = m2  # sum of squared deviations from the current mean

    def update(self, value: float) -> None:
        """Add one observation."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        """Population variance. Returns 0.0 if count < 2."""
        if self.count < 2:
            return 0.0
        return self.m2 / self.count

    @property
    def std(self) -> float:
        """Population standard deviation."""
        return math.sqrt(self.variance)

    def to_dict(self) -> dict[str, float]:
        """Serialize to dict for JSON persistence."""
        return {"count": self.count, "mean": self.mean, "m2": self.m2}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> "WelfordAccumulator":
        """Restore from serialized dict."""
        return cls(count=int(d["count"]), mean=float(d["mean"]), m2=float(d["m2"]))

    def merge(self, other: "WelfordAccumulator") -> "WelfordAccumulator":
        """Merge two accumulators (parallel Welford).

        Returns a new accumulator representing the combined stream.
        Neither self nor other is modified.

        Reference: Chan et al. (1979), "Updating Formulae and a Pairwise
        Algorithm for Computing Sample Variances."
        """
        if other.count == 0:
            return WelfordAccumulator(self.count, self.mean, self.m2)
        if self.count == 0:
            return WelfordAccumulator(other.count, other.mean, other.m2)

        combined_count = self.count + other.count
        delta = other.mean - self.mean
        combined_mean = (self.mean * self.count + other.mean * other.count) / combined_count
        combined_m2 = (
            self.m2 + other.m2
            + delta * delta * self.count * other.count / combined_count
        )
        return WelfordAccumulator(combined_count, combined_mean, combined_m2)


# ---------------------------------------------------------------------------
# Welford covariance accumulator
# ---------------------------------------------------------------------------

class WelfordCovarianceAccumulator:
    """Online covariance between two streams using the Welford/parallel method.

    Tracks mean_x, mean_y, and the co-moment C = sum((xi - mean_x)(yi - mean_y)).
    Covariance = C / count (population covariance).

    This allows measuring whether low-loss regions correspond to
    high-accuracy regions, per index combination.
    """
    __slots__ = ("count", "mean_x", "mean_y", "co_moment")

    def __init__(
        self,
        count: int = 0,
        mean_x: float = 0.0,
        mean_y: float = 0.0,
        co_moment: float = 0.0,
    ):
        self.count = count
        self.mean_x = mean_x
        self.mean_y = mean_y
        self.co_moment = co_moment

    def update(self, x: float, y: float) -> None:
        """Add one paired observation (x, y).

        Uses the stable one-pass co-moment formula from Pebay (2008):
        After computing delta_x = x - mean_x_old, update mean_x, then
        update mean_y, then accumulate co_moment using delta_x and the
        NEW mean_y: co_moment += delta_x * (y - mean_y_new).
        """
        self.count += 1
        dx = x - self.mean_x
        self.mean_x += dx / self.count
        dy = y - self.mean_y
        self.mean_y += dy / self.count
        # co_moment uses delta_x (old mean_x) and (y - mean_y_new)
        self.co_moment += dx * (y - self.mean_y)

    @property
    def covariance(self) -> float:
        """Population covariance. Returns 0.0 if count < 2."""
        if self.count < 2:
            return 0.0
        return self.co_moment / self.count

    def to_dict(self) -> dict[str, float]:
        return {
            "count": self.count,
            "mean_x": self.mean_x,
            "mean_y": self.mean_y,
            "co_moment": self.co_moment,
        }

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> "WelfordCovarianceAccumulator":
        return cls(
            count=int(d["count"]),
            mean_x=float(d["mean_x"]),
            mean_y=float(d["mean_y"]),
            co_moment=float(d["co_moment"]),
        )


# ---------------------------------------------------------------------------
# Per-index-combination cell
# ---------------------------------------------------------------------------

class _MetricsCell:
    """Statistics cell for one unique index combination.

    Tracks:
      - Accuracy (Welford: mean accuracy, variance of accuracy)
      - Loss (Welford: mean loss, variance of loss)
      - Covariance between accuracy and loss
      - Raw score statistics (preferred and rejected score means)
    """
    __slots__ = (
        "accuracy", "loss", "acc_loss_cov",
        "score_preferred", "score_rejected",
    )

    def __init__(self):
        self.accuracy = WelfordAccumulator()
        self.loss = WelfordAccumulator()
        self.acc_loss_cov = WelfordCovarianceAccumulator()
        self.score_preferred = WelfordAccumulator()
        self.score_rejected = WelfordAccumulator()

    def update(
        self,
        correct: float,
        loss_val: float,
        score_pref: float,
        score_rej: float,
    ) -> None:
        """Record one pair result."""
        self.accuracy.update(correct)
        self.loss.update(loss_val)
        self.acc_loss_cov.update(correct, loss_val)
        self.score_preferred.update(score_pref)
        self.score_rejected.update(score_rej)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy.to_dict(),
            "loss": self.loss.to_dict(),
            "acc_loss_cov": self.acc_loss_cov.to_dict(),
            "score_preferred": self.score_preferred.to_dict(),
            "score_rejected": self.score_rejected.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "_MetricsCell":
        cell = cls()
        cell.accuracy = WelfordAccumulator.from_dict(d["accuracy"])
        cell.loss = WelfordAccumulator.from_dict(d["loss"])
        cell.acc_loss_cov = WelfordCovarianceAccumulator.from_dict(d["acc_loss_cov"])
        cell.score_preferred = WelfordAccumulator.from_dict(d["score_preferred"])
        cell.score_rejected = WelfordAccumulator.from_dict(d["score_rejected"])
        return cell

    def summary_dict(self) -> dict[str, float]:
        """Flat summary for logging."""
        return {
            "count": self.accuracy.count,
            "accuracy_mean": self.accuracy.mean,
            "accuracy_std": self.accuracy.std,
            "loss_mean": self.loss.mean,
            "loss_std": self.loss.std,
            "acc_loss_cov": self.acc_loss_cov.covariance,
            "score_pref_mean": self.score_preferred.mean,
            "score_rej_mean": self.score_rejected.mean,
        }


# ---------------------------------------------------------------------------
# PairResult: the update payload
# ---------------------------------------------------------------------------

class PairResult:
    """Data class for one scored pair result.

    This is the input to ValidationMetrics.update(). The training loop
    constructs one PairResult per (head, pair) combination and passes it
    to the tracker.

    All fields are plain Python types (no tensors). The training loop is
    responsible for .item() conversion before constructing this.
    """
    __slots__ = (
        "head_name",
        "correct", "loss_contribution",
        "score_preferred", "score_rejected",
        "width_a", "height_a", "sigma_a",
        "width_b", "height_b", "sigma_b",
        "aspect_ratio_a", "aspect_ratio_b",
        "source_a", "source_b",
        "traj_a", "step_a",
        "traj_b", "step_b",
    )

    def __init__(
        self,
        head_name: str,
        correct: float,
        loss_contribution: float,
        score_preferred: float,
        score_rejected: float,
        width_a: int = 1280,
        height_a: int = 832,
        sigma_a: float = 0.0,
        width_b: int = 1280,
        height_b: int = 832,
        sigma_b: float = 0.0,
        source_a: str = "unknown",
        source_b: str = "unknown",
        traj_a: int = -1,
        step_a: str = "",
        traj_b: int = -1,
        step_b: str = "",
    ):
        self.head_name = head_name
        self.correct = correct  # 1.0 if model agreed with ground truth, else 0.0
        self.loss_contribution = loss_contribution
        self.score_preferred = score_preferred
        self.score_rejected = score_rejected
        self.width_a = width_a
        self.height_a = height_a
        self.sigma_a = sigma_a
        self.width_b = width_b
        self.height_b = height_b
        self.sigma_b = sigma_b
        self.aspect_ratio_a = width_a / max(height_a, 1)
        self.aspect_ratio_b = width_b / max(height_b, 1)
        self.source_a = source_a
        self.source_b = source_b
        self.traj_a = traj_a
        self.step_a = step_a
        self.traj_b = traj_b
        self.step_b = step_b


# ---------------------------------------------------------------------------
# ValidationMetrics: the multi-indexed tracker
# ---------------------------------------------------------------------------

class ValidationMetrics:
    """Multi-indexed accuracy/loss tracker for BTRM training.

    Maintains running statistics (Welford's online algorithm) per unique
    combination of indexing dimensions:
      - head_name (e.g., "pinkify", "thisnotthat")
      - resolution_bucket (megapixel bucket of the image pair)
      - logsnr_bucket (sigma/logSNR bucket of each image)
      - aspect_bucket (portrait/square/landscape)
      - source (trajectory provenance label)

    Each pair contributes to multiple index combinations. For a pair with
    images A (512x512, sigma=0.034, gpu_rollout) and B (512x512, sigma=0.5,
    original_v1), the result is indexed under:
      - ("pinkify", "0.2-0.4 MP", *, *, *)
      - ("pinkify", *, "near-clean", *, *)  -- from sigma_a
      - ("pinkify", *, "moderate", *, *)    -- from sigma_b
      - ("pinkify", *, *, "square", *)
      - ("pinkify", *, *, *, "gpu_rollout")
      - ("pinkify", *, *, *, "original_v1")
      - and all specific cross-indexed combinations

    The tracker does NOT accumulate raw values. Memory is O(number of
    unique index combinations), not O(number of pairs seen). For typical
    configurations this is ~200-500 cells.

    Usage:
        tracker = ValidationMetrics()

        # In the training loop, after scoring a pair:
        tracker.update(PairResult(
            head_name="pinkify",
            correct=1.0,
            loss_contribution=0.45,
            score_preferred=2.3,
            score_rejected=1.1,
            width_a=512, height_a=512, sigma_a=0.034,
            width_b=512, height_b=512, sigma_b=0.5,
            source_a="gpu_rollout",
            source_b="original_v1",
        ))

        # Periodically, log a summary:
        summary = tracker.summary()
        # summary is a flat dict suitable for JSONL

        # Persist full state:
        tracker.save_json("validation_metrics.json")

        # Restore:
        tracker = ValidationMetrics.load_json("validation_metrics.json")
    """

    def __init__(self):
        # Multi-indexed cells: key is a tuple-string of index values.
        # We use several dictionaries for different index granularities.

        # Single-axis cells: one dimension at a time
        self._by_head: dict[str, _MetricsCell] = {}
        self._by_resolution: dict[str, _MetricsCell] = {}
        self._by_logsnr: dict[str, _MetricsCell] = {}
        self._by_aspect: dict[str, _MetricsCell] = {}
        self._by_source: dict[str, _MetricsCell] = {}

        # Cross-indexed cells: head x resolution (the primary funfetti question)
        self._by_head_resolution: dict[str, _MetricsCell] = {}

        # Cross-indexed: head x logsnr
        self._by_head_logsnr: dict[str, _MetricsCell] = {}

        # Cross-indexed: resolution x logsnr (for the funfetti NFE question)
        self._by_resolution_logsnr: dict[str, _MetricsCell] = {}

        # Global accumulator (all pairs, all heads)
        self._global = _MetricsCell()

        # Step counter (how many update() calls total)
        self._n_updates = 0

    def _get_or_create(
        self, store: dict[str, _MetricsCell], key: str
    ) -> _MetricsCell:
        """Get or create a cell in a given store."""
        if key not in store:
            store[key] = _MetricsCell()
        return store[key]

    def update(self, result: PairResult) -> None:
        """Record one pair result across all relevant index combinations.

        For each pair, both images contribute resolution/logsnr/source
        indices. The pair result is recorded under both image A's and
        image B's index values, since the pair's accuracy reflects the
        model's ability to discriminate at *both* resolutions and noise
        levels involved.

        Args:
            result: A PairResult with all metadata fields populated.
        """
        self._n_updates += 1

        correct = result.correct
        loss = result.loss_contribution
        s_pref = result.score_preferred
        s_rej = result.score_rejected

        # --- Global ---
        self._global.update(correct, loss, s_pref, s_rej)

        # --- By head ---
        self._get_or_create(self._by_head, result.head_name).update(
            correct, loss, s_pref, s_rej
        )

        # --- Resolution buckets (both images) ---
        res_a = resolution_bucket(result.width_a, result.height_a)
        res_b = resolution_bucket(result.width_b, result.height_b)
        for res in {res_a, res_b}:  # set avoids double-counting same bucket
            self._get_or_create(self._by_resolution, res).update(
                correct, loss, s_pref, s_rej
            )

        # --- LogSNR buckets (both images) ---
        lsnr_a = logsnr_bucket(result.sigma_a)
        lsnr_b = logsnr_bucket(result.sigma_b)
        for lsnr in {lsnr_a, lsnr_b}:
            self._get_or_create(self._by_logsnr, lsnr).update(
                correct, loss, s_pref, s_rej
            )

        # --- Aspect ratio buckets (both images) ---
        asp_a = aspect_bucket(result.width_a, result.height_a)
        asp_b = aspect_bucket(result.width_b, result.height_b)
        for asp in {asp_a, asp_b}:
            self._get_or_create(self._by_aspect, asp).update(
                correct, loss, s_pref, s_rej
            )

        # --- Source (both images) ---
        for src in {result.source_a, result.source_b}:
            self._get_or_create(self._by_source, src).update(
                correct, loss, s_pref, s_rej
            )

        # --- Cross-indexed: head x resolution ---
        for res in {res_a, res_b}:
            key = f"{result.head_name}|{res}"
            self._get_or_create(self._by_head_resolution, key).update(
                correct, loss, s_pref, s_rej
            )

        # --- Cross-indexed: head x logsnr ---
        for lsnr in {lsnr_a, lsnr_b}:
            key = f"{result.head_name}|{lsnr}"
            self._get_or_create(self._by_head_logsnr, key).update(
                correct, loss, s_pref, s_rej
            )

        # --- Cross-indexed: resolution x logsnr ---
        for res in {res_a, res_b}:
            for lsnr in {lsnr_a, lsnr_b}:
                key = f"{res}|{lsnr}"
                self._get_or_create(self._by_resolution_logsnr, key).update(
                    correct, loss, s_pref, s_rej
                )

    def query(
        self,
        head: str | None = None,
        resolution: str | None = None,
        logsnr: str | None = None,
        aspect: str | None = None,
        source: str | None = None,
    ) -> dict[str, float] | None:
        """Query statistics for a specific index or combination.

        Supports single-axis and two-axis cross queries. For single-axis
        queries, pass one argument. For cross queries, pass two. Returns
        None if the requested combination has no data.

        Args:
            head: Head name (e.g., "pinkify").
            resolution: Resolution bucket label (e.g., "0.2-0.4 MP").
            logsnr: LogSNR bucket label.
            aspect: Aspect ratio bucket label.
            source: Trajectory source label.

        Returns:
            Summary dict with count, accuracy_mean, loss_mean, etc.,
            or None if the combination has no observations.
        """
        # Count how many axes are specified
        specified = sum(
            1 for v in [head, resolution, logsnr, aspect, source]
            if v is not None
        )

        if specified == 0:
            return self._global.summary_dict()

        if specified == 1:
            if head is not None:
                cell = self._by_head.get(head)
            elif resolution is not None:
                cell = self._by_resolution.get(resolution)
            elif logsnr is not None:
                cell = self._by_logsnr.get(logsnr)
            elif aspect is not None:
                cell = self._by_aspect.get(aspect)
            else:
                cell = self._by_source.get(source)
            return cell.summary_dict() if cell else None

        if specified == 2:
            if head is not None and resolution is not None:
                key = f"{head}|{resolution}"
                cell = self._by_head_resolution.get(key)
            elif head is not None and logsnr is not None:
                key = f"{head}|{logsnr}"
                cell = self._by_head_logsnr.get(key)
            elif resolution is not None and logsnr is not None:
                key = f"{resolution}|{logsnr}"
                cell = self._by_resolution_logsnr.get(key)
            else:
                # Cross-indices beyond the pre-computed ones: not available.
                return None
            return cell.summary_dict() if cell else None

        # 3+ axis queries not pre-computed
        return None

    def summary(self) -> dict[str, Any]:
        """Return a flat summary dict suitable for JSONL logging.

        The dict contains:
          - "n_updates": total pair results recorded
          - "global_accuracy": overall mean accuracy
          - "global_loss": overall mean loss
          - Per-head accuracy: "accuracy_{head_name}" for each head
          - Per-resolution accuracy: "accuracy_{bucket}" for each bucket
          - Per-logsnr accuracy: "accuracy_logsnr_{bucket}" for each bucket
          - Sample counts: "n_{head_name}", "n_{bucket}", etc.

        This is designed to be appended to the training metrics JSONL
        alongside existing per-step fields (loss, bt_loss, grad_norm, etc.).
        """
        out: dict[str, Any] = {
            "n_updates": self._n_updates,
            "global_accuracy": self._global.accuracy.mean,
            "global_loss": self._global.loss.mean,
            "global_count": self._global.accuracy.count,
        }

        # Per-head summary
        for head_name, cell in sorted(self._by_head.items()):
            safe = head_name.replace(" ", "_")
            out[f"val_accuracy_{safe}"] = cell.accuracy.mean
            out[f"val_loss_{safe}"] = cell.loss.mean
            out[f"val_n_{safe}"] = cell.accuracy.count

        # Per-resolution summary
        for bucket, cell in sorted(self._by_resolution.items()):
            safe = bucket.replace(" ", "_").replace(".", "p")
            out[f"val_accuracy_res_{safe}"] = cell.accuracy.mean
            out[f"val_n_res_{safe}"] = cell.accuracy.count

        # Per-logsnr summary
        for bucket, cell in sorted(self._by_logsnr.items()):
            safe = _sanitize_key(bucket)
            out[f"val_accuracy_logsnr_{safe}"] = cell.accuracy.mean
            out[f"val_n_logsnr_{safe}"] = cell.accuracy.count

        # Per-source summary
        for source, cell in sorted(self._by_source.items()):
            safe = source.replace(" ", "_")
            out[f"val_accuracy_src_{safe}"] = cell.accuracy.mean
            out[f"val_n_src_{safe}"] = cell.accuracy.count

        return out

    # ----- Serialization -----

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full tracker state to a dict for JSON persistence.

        The dict is designed for lossless round-tripping: from_dict(to_dict())
        recovers the exact same accumulator state (count, mean, m2).
        """
        return {
            "version": 1,
            "n_updates": self._n_updates,
            "global": self._global.to_dict(),
            "by_head": {k: v.to_dict() for k, v in self._by_head.items()},
            "by_resolution": {k: v.to_dict() for k, v in self._by_resolution.items()},
            "by_logsnr": {k: v.to_dict() for k, v in self._by_logsnr.items()},
            "by_aspect": {k: v.to_dict() for k, v in self._by_aspect.items()},
            "by_source": {k: v.to_dict() for k, v in self._by_source.items()},
            "by_head_resolution": {k: v.to_dict() for k, v in self._by_head_resolution.items()},
            "by_head_logsnr": {k: v.to_dict() for k, v in self._by_head_logsnr.items()},
            "by_resolution_logsnr": {k: v.to_dict() for k, v in self._by_resolution_logsnr.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ValidationMetrics":
        """Restore tracker state from a serialized dict."""
        vm = cls()
        vm._n_updates = d.get("n_updates", 0)
        vm._global = _MetricsCell.from_dict(d["global"])

        for store_name in [
            "by_head", "by_resolution", "by_logsnr", "by_aspect",
            "by_source", "by_head_resolution", "by_head_logsnr",
            "by_resolution_logsnr",
        ]:
            store = getattr(vm, f"_{store_name}")
            for k, v in d.get(store_name, {}).items():
                store[k] = _MetricsCell.from_dict(v)

        return vm

    def save_json(self, path: str) -> None:
        """Persist full state to a JSON file.

        Overwrites the file atomically (write to temp, then rename) to
        prevent corruption if the process is interrupted.
        """
        import os
        import tempfile

        data = self.to_dict()
        dir_name = os.path.dirname(os.path.abspath(path))

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json.tmp", dir=dir_name, delete=False
        ) as f:
            tmp_path = f.name
            json.dump(data, f, indent=2)

        os.replace(tmp_path, path)

    @classmethod
    def load_json(cls, path: str) -> "ValidationMetrics":
        """Restore tracker from a JSON file.

        Args:
            path: Path to the JSON file written by save_json().

        Returns:
            ValidationMetrics instance with restored state.
        """
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    # ----- Tabular display -----

    def format_table(
        self,
        axis: str = "resolution",
        head: str | None = None,
    ) -> str:
        """Format a human-readable table for one index axis.

        Args:
            axis: "resolution", "logsnr", "head", "source", or "aspect".
            head: If not None and axis is "resolution" or "logsnr", show
                the cross-indexed head x axis breakdown.

        Returns:
            Multi-line string with aligned columns.
        """
        lines = []

        if head is not None and axis == "resolution":
            store = self._by_head_resolution
            prefix = f"{head}|"
        elif head is not None and axis == "logsnr":
            store = self._by_head_logsnr
            prefix = f"{head}|"
        else:
            store = getattr(self, f"_by_{axis}", {})
            prefix = ""

        # Header
        lines.append(
            f"{'Bucket':<40s} {'Count':>8s} {'Accuracy':>10s} "
            f"{'Acc Std':>10s} {'Loss':>10s} {'Loss Std':>10s} "
            f"{'Acc-Loss Cov':>12s}"
        )
        lines.append("-" * 100)

        for key in sorted(store.keys()):
            if prefix and not key.startswith(prefix):
                continue
            cell = store[key]
            label = key[len(prefix):] if prefix else key
            s = cell.summary_dict()
            lines.append(
                f"{label:<40s} {s['count']:>8d} {s['accuracy_mean']:>10.4f} "
                f"{s['accuracy_std']:>10.4f} {s['loss_mean']:>10.4f} "
                f"{s['loss_std']:>10.4f} {s['acc_loss_cov']:>12.6f}"
            )

        # Total row
        g = self._global.summary_dict()
        lines.append("-" * 100)
        lines.append(
            f"{'TOTAL':<40s} {g['count']:>8d} {g['accuracy_mean']:>10.4f} "
            f"{g['accuracy_std']:>10.4f} {g['loss_mean']:>10.4f} "
            f"{g['loss_std']:>10.4f} {g['acc_loss_cov']:>12.6f}"
        )

        return "\n".join(lines)

    @property
    def resolution_buckets_seen(self) -> list[str]:
        """List of resolution buckets that have received at least one update."""
        return sorted(self._by_resolution.keys())

    @property
    def logsnr_buckets_seen(self) -> list[str]:
        """List of logSNR buckets that have received at least one update."""
        return sorted(self._by_logsnr.keys())

    @property
    def heads_seen(self) -> list[str]:
        """List of head names that have received at least one update."""
        return sorted(self._by_head.keys())

    @property
    def sources_seen(self) -> list[str]:
        """List of trajectory sources that have received at least one update."""
        return sorted(self._by_source.keys())


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _sanitize_key(s: str) -> str:
    """Sanitize a bucket label for use as a JSON/JSONL key fragment.

    Replaces spaces, parens, angle brackets, and dots with safe characters.
    """
    return (
        s.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("<", "lt")
        .replace(">", "gt")
        .replace(".", "p")
        .replace("=", "eq")
        .replace("<=", "leq")
    )

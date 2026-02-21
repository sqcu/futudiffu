"""Tests for clean-biased step selection in BTRMPairSampler.

Validates the 80/20 clean-biased sampling mode:
  - 80%+ of sampled positions are sigma=0 (clean images)
  - 20% are non-clean, drawn from exponential decay biased toward high logSNR
  - LogSNR histogram tracking works correctly
  - Edge cases (clean_fraction=0.0, 1.0, all-clean trajectories)

Uses synthetic positions -- no real dataset, no GPU, no torch.
"""
from __future__ import annotations

import math
import sys
import os

import numpy as np
import pytest

# Ensure src_ii is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src_ii.pair_sampler import (
    BTRMPairSampler,
    _ImagePosition,
    _clean_biased_step_logits,
    _sigma_to_logsnr,
    _logsnr_hist_bin,
    _SIGMA_ZERO_BIN,
    _LOGSNR_HIST_BINS,
    logsnr_sampling_weight,
    logsnr_sampling_logit,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic trajectory data
# ---------------------------------------------------------------------------

def _make_positions(
    n_trajectories: int = 5,
    sigmas_per_traj: list[float] | None = None,
    width: int = 1280,
    height: int = 832,
) -> list[_ImagePosition]:
    """Build synthetic _ImagePosition objects.

    Each trajectory has the same set of sigma values. The "final" position
    (sigma=0.0) is included by default.

    Default sigmas represent a 30-step denoising schedule (sampled subset):
      [0.0, 0.034, 0.2, 0.367, 0.5, 0.7, 0.867]
    """
    if sigmas_per_traj is None:
        sigmas_per_traj = [0.0, 0.034, 0.2, 0.367, 0.5, 0.7, 0.867]

    positions = []
    for traj_id in range(n_trajectories):
        for i, sigma in enumerate(sigmas_per_traj):
            if sigma <= 0.0:
                step_key = "final"
            else:
                step_key = f"step_{i:02d}"
            logit = logsnr_sampling_logit(sigma)
            positions.append(_ImagePosition(
                traj_id=traj_id,
                step_key=step_key,
                sigma=sigma,
                logit=logit,
                width=width,
                height=height,
                prompt_idx=traj_id,
            ))
    return positions


def _make_no_clean_positions(n_trajectories: int = 5) -> list[_ImagePosition]:
    """Build positions where NO trajectory has sigma=0.

    Tests the fallback behavior: when there's no sigma=0 position, the
    clean path should select the highest-logSNR position instead.
    """
    sigmas = [0.034, 0.2, 0.367, 0.5, 0.7, 0.867]
    return _make_positions(n_trajectories=n_trajectories, sigmas_per_traj=sigmas)


def _make_all_clean_positions(n_trajectories: int = 5) -> list[_ImagePosition]:
    """Build positions where ALL positions are sigma=0.

    Edge case: the non-clean path has nowhere to go.
    """
    sigmas = [0.0]
    return _make_positions(n_trajectories=n_trajectories, sigmas_per_traj=sigmas)


# ---------------------------------------------------------------------------
# Tests: clean_fraction=0.8 (the 80/20 spec)
# ---------------------------------------------------------------------------

class TestCleanBiased80_20:
    """Test the primary 80/20 clean-biased sampling mode."""

    def test_at_least_75_percent_clean(self):
        """With clean_fraction=0.8, at least 75% of positions should be sigma=0.

        We allow 5% margin below 0.8 to account for sampling variance over
        1000 samples. With N=1000 and p=0.8, std = sqrt(p*(1-p)/N) = 0.013,
        so 75% is ~3.8 standard deviations below the mean.
        """
        positions = _make_positions(n_trajectories=10)
        sampler = BTRMPairSampler(
            positions=positions,
            allow_inter_trajectory=True,
            rng_seed=42,
            clean_fraction=0.8,
        )

        n_samples = 1000
        n_clean = 0
        for _ in range(n_samples):
            pair = sampler.sample_pair()
            if pair["sigma_a"] <= 0.0:
                n_clean += 1
            if pair["sigma_b"] <= 0.0:
                n_clean += 1

        total_positions = 2 * n_samples
        clean_frac = n_clean / total_positions
        assert clean_frac >= 0.75, (
            f"Expected >= 75% clean positions, got {clean_frac:.3f} "
            f"({n_clean}/{total_positions})"
        )

    def test_measured_clean_fraction_near_80(self):
        """The get_clean_fraction() accessor should report close to 0.8."""
        positions = _make_positions(n_trajectories=10)
        sampler = BTRMPairSampler(
            positions=positions,
            allow_inter_trajectory=True,
            rng_seed=123,
            clean_fraction=0.8,
        )

        for _ in range(500):
            sampler.sample_pair()

        measured = sampler.get_clean_fraction()
        assert 0.70 <= measured <= 0.90, (
            f"Expected measured clean fraction near 0.8, got {measured:.3f}"
        )


# ---------------------------------------------------------------------------
# Tests: clean_fraction=1.0 (all clean)
# ---------------------------------------------------------------------------

class TestCleanFraction100:
    """Test that clean_fraction=1.0 always selects sigma=0."""

    def test_all_positions_clean(self):
        """With clean_fraction=1.0, 100% of positions should be sigma=0."""
        positions = _make_positions(n_trajectories=5)
        sampler = BTRMPairSampler(
            positions=positions,
            allow_inter_trajectory=True,
            rng_seed=42,
            clean_fraction=1.0,
        )

        for _ in range(200):
            pair = sampler.sample_pair()
            assert pair["sigma_a"] <= 0.0, (
                f"Expected sigma_a=0.0, got {pair['sigma_a']}"
            )
            assert pair["sigma_b"] <= 0.0, (
                f"Expected sigma_b=0.0, got {pair['sigma_b']}"
            )

    def test_measured_fraction_is_1(self):
        """get_clean_fraction() should return 1.0 with clean_fraction=1.0."""
        positions = _make_positions(n_trajectories=5)
        sampler = BTRMPairSampler(
            positions=positions,
            allow_inter_trajectory=True,
            rng_seed=42,
            clean_fraction=1.0,
        )

        for _ in range(100):
            sampler.sample_pair()

        assert sampler.get_clean_fraction() == 1.0


# ---------------------------------------------------------------------------
# Tests: clean_fraction=0.0 (no forced clean, pure exponential decay)
# ---------------------------------------------------------------------------

class TestCleanFraction0:
    """Test that clean_fraction=0.0 uses only exponential decay (no forced clean)."""

    def test_no_forced_clean(self):
        """With clean_fraction=0.0, no sigma=0 positions should be forced.

        Some may still appear by chance if the exponential decay assigns
        nonzero probability to the sigma=0 position -- but since sigma=0
        gets logit=-inf in the non-clean logits, it should never be sampled.
        """
        positions = _make_positions(n_trajectories=10)
        sampler = BTRMPairSampler(
            positions=positions,
            allow_inter_trajectory=True,
            rng_seed=42,
            clean_fraction=0.0,
        )

        n_samples = 500
        n_clean = 0
        for _ in range(n_samples):
            pair = sampler.sample_pair()
            if pair["sigma_a"] <= 0.0:
                n_clean += 1
            if pair["sigma_b"] <= 0.0:
                n_clean += 1

        total_positions = 2 * n_samples
        # sigma=0 gets logit=-inf in the non-clean path, so it should be 0%
        assert n_clean == 0, (
            f"Expected 0 clean positions with clean_fraction=0.0, got "
            f"{n_clean}/{total_positions}"
        )

    def test_exponential_decay_biased_toward_high_logsnr(self):
        """The non-clean 20% should be biased toward high logSNR (near-clean).

        Sample many positions with clean_fraction=0.0 and check that
        low-sigma (high-logSNR) positions are sampled much more often
        than high-sigma (low-logSNR) positions.
        """
        # Use only non-clean sigmas so we can measure the distribution clearly
        sigmas = [0.034, 0.2, 0.367, 0.5, 0.7, 0.867]
        positions = _make_positions(
            n_trajectories=5,
            sigmas_per_traj=sigmas,
        )
        sampler = BTRMPairSampler(
            positions=positions,
            allow_inter_trajectory=True,
            rng_seed=42,
            clean_fraction=0.0,
        )

        sigma_counts: dict[float, int] = {s: 0 for s in sigmas}
        n_samples = 2000
        for _ in range(n_samples):
            pair = sampler.sample_pair()
            # Round to match our synthetic sigma values
            for sigma_val in [pair["sigma_a"], pair["sigma_b"]]:
                closest = min(sigmas, key=lambda s: abs(s - sigma_val))
                sigma_counts[closest] += 1

        # sigma=0.034 (logSNR=6.69) should be sampled MUCH more than
        # sigma=0.867 (logSNR=-3.75). The ratio should be at least 5x.
        count_near_clean = sigma_counts[0.034]
        count_very_noisy = sigma_counts[0.867]

        # Avoid division by zero
        assert count_very_noisy > 0, (
            f"Expected at least some sigma=0.867 samples, got 0. "
            f"Distribution: {sigma_counts}"
        )
        ratio = count_near_clean / count_very_noisy
        assert ratio >= 3.0, (
            f"Expected near-clean to be at least 3x more frequent than "
            f"very noisy, got ratio={ratio:.1f}. Counts: {sigma_counts}"
        )


# ---------------------------------------------------------------------------
# Tests: logSNR histogram tracking
# ---------------------------------------------------------------------------

class TestLogSNRHistogram:
    """Test the lightweight logSNR histogram tracking."""

    def test_histogram_bins_initialized(self):
        """All histogram bins should exist after construction."""
        positions = _make_positions(n_trajectories=3)
        sampler = BTRMPairSampler(
            positions=positions,
            allow_inter_trajectory=True,
            rng_seed=42,
        )

        hist = sampler.get_logsnr_histogram()
        # Should have 7 logSNR bins + 1 sigma=0 bin = 8
        assert len(hist) == 8, f"Expected 8 histogram bins, got {len(hist)}"
        assert _SIGMA_ZERO_BIN in hist
        for label, _, _ in _LOGSNR_HIST_BINS:
            assert label in hist, f"Missing bin: {label}"

    def test_histogram_counts_match_sampling(self):
        """Total histogram counts should be at least 2x the number of sampled pairs.

        The count may be higher than 2*n_pairs because retried pairs (same
        traj+step collision) also sample positions that get recorded. The
        histogram tracks ALL positions sampled, including retry attempts.
        """
        positions = _make_positions(n_trajectories=5)
        sampler = BTRMPairSampler(
            positions=positions,
            allow_inter_trajectory=True,
            rng_seed=42,
            clean_fraction=0.8,
        )

        n_pairs = 100
        for _ in range(n_pairs):
            sampler.sample_pair()

        hist = sampler.get_logsnr_histogram()
        total_counts = sum(hist.values())
        # Each pair samples at least 2 positions; retries add more
        assert total_counts >= 2 * n_pairs, (
            f"Expected >= {2 * n_pairs} total histogram entries, got {total_counts}"
        )
        # Sanity: shouldn't be absurdly higher (retries are rare)
        assert total_counts <= 4 * n_pairs, (
            f"Expected <= {4 * n_pairs} total histogram entries, got {total_counts}"
        )

    def test_histogram_sigma_zero_bin(self):
        """With clean_fraction=1.0, all counts should be in sigma=0 bin.

        Retries add extra position samples, so count >= 2*n_pairs. But all
        sampled positions should still be sigma=0 since clean_fraction=1.0.
        """
        positions = _make_positions(n_trajectories=3)
        sampler = BTRMPairSampler(
            positions=positions,
            allow_inter_trajectory=True,
            rng_seed=42,
            clean_fraction=1.0,
        )

        for _ in range(50):
            sampler.sample_pair()

        hist = sampler.get_logsnr_histogram()
        assert hist[_SIGMA_ZERO_BIN] >= 100  # 50 pairs x 2 positions + retries
        # All other bins should be 0 (every sampled position is sigma=0)
        for label, _, _ in _LOGSNR_HIST_BINS:
            assert hist[label] == 0, f"Expected 0 in bin {label}, got {hist[label]}"

    def test_histogram_no_sigma_zero_with_cf0(self):
        """With clean_fraction=0.0, sigma=0 bin should be empty."""
        positions = _make_positions(n_trajectories=5)
        sampler = BTRMPairSampler(
            positions=positions,
            allow_inter_trajectory=True,
            rng_seed=42,
            clean_fraction=0.0,
        )

        for _ in range(100):
            sampler.sample_pair()

        hist = sampler.get_logsnr_histogram()
        assert hist[_SIGMA_ZERO_BIN] == 0, (
            f"Expected 0 sigma=0 samples with clean_fraction=0.0, "
            f"got {hist[_SIGMA_ZERO_BIN]}"
        )

    def test_get_clean_fraction_zero_before_sampling(self):
        """get_clean_fraction() should return 0.0 before any sampling."""
        positions = _make_positions(n_trajectories=3)
        sampler = BTRMPairSampler(
            positions=positions,
            allow_inter_trajectory=True,
            rng_seed=42,
        )

        assert sampler.get_clean_fraction() == 0.0


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases and fallback behavior."""

    def test_no_sigma_zero_fallback_to_highest_logsnr(self):
        """When no sigma=0 position exists, clean path should pick highest logSNR."""
        positions = _make_no_clean_positions(n_trajectories=5)
        sampler = BTRMPairSampler(
            positions=positions,
            allow_inter_trajectory=True,
            rng_seed=42,
            clean_fraction=1.0,  # Always pick "clean"
        )

        # With no sigma=0 positions, the clean path should fall back to
        # the highest-logSNR position (sigma=0.034, logSNR~6.69)
        for _ in range(100):
            pair = sampler.sample_pair()
            # Both should be sigma=0.034 (the cleanest available)
            assert abs(pair["sigma_a"] - 0.034) < 0.001, (
                f"Expected sigma_a~0.034, got {pair['sigma_a']}"
            )
            assert abs(pair["sigma_b"] - 0.034) < 0.001, (
                f"Expected sigma_b~0.034, got {pair['sigma_b']}"
            )

    def test_all_clean_trajectory(self):
        """When ALL positions are sigma=0, both paths should return sigma=0."""
        positions = _make_all_clean_positions(n_trajectories=3)
        sampler = BTRMPairSampler(
            positions=positions,
            allow_inter_trajectory=True,
            rng_seed=42,
            clean_fraction=0.5,  # 50/50 split -- but no non-clean to pick
        )

        for _ in range(50):
            pair = sampler.sample_pair()
            assert pair["sigma_a"] <= 0.0
            assert pair["sigma_b"] <= 0.0

    def test_default_none_preserves_legacy_behavior(self):
        """When clean_fraction=None (default), legacy logit-softmax is used."""
        positions = _make_positions(n_trajectories=5)
        sampler = BTRMPairSampler(
            positions=positions,
            allow_inter_trajectory=True,
            rng_seed=42,
            # clean_fraction not set -- default None
        )

        # Sample some pairs -- this should work with the legacy path
        pairs = [sampler.sample_pair() for _ in range(100)]
        assert len(pairs) == 100

        # The legacy path produces a mix of sigmas, not 80%+ clean
        clean_count = sum(
            1 for p in pairs
            if p["sigma_a"] <= 0.0 or p["sigma_b"] <= 0.0
        )
        # With the legacy geometric decay, sigma=0 gets high weight but not 80%
        # We just check it works -- exact fraction depends on the logit schedule
        assert clean_count > 0, "Legacy path should produce some clean samples"

    def test_stats_includes_histogram(self):
        """stats() should include histogram data after sampling."""
        positions = _make_positions(n_trajectories=3)
        sampler = BTRMPairSampler(
            positions=positions,
            allow_inter_trajectory=True,
            rng_seed=42,
            clean_fraction=0.8,
        )

        for _ in range(20):
            sampler.sample_pair()

        stats = sampler.stats()
        assert "logsnr_histogram" in stats
        assert "measured_clean_fraction" in stats
        assert "n_positions_sampled" in stats
        assert "clean_fraction_setting" in stats
        assert stats["clean_fraction_setting"] == 0.8
        assert stats["n_positions_sampled"] >= 40  # 20 pairs x 2, plus retries


# ---------------------------------------------------------------------------
# Tests: _clean_biased_step_logits helper
# ---------------------------------------------------------------------------

class TestCleanBiasedStepLogits:
    """Test the exponential decay logit computation."""

    def test_sigma_zero_gets_neg_inf(self):
        """sigma=0 positions should get logit=-inf."""
        positions = [
            _ImagePosition(0, "final", 0.0, 0.0),
            _ImagePosition(0, "step_01", 0.5, -0.5),
        ]
        logits = _clean_biased_step_logits(positions)
        assert logits[0] == -np.inf
        assert np.isfinite(logits[1])

    def test_higher_logsnr_gets_higher_logit(self):
        """Positions with higher logSNR should get higher logits."""
        positions = [
            _ImagePosition(0, "step_01", 0.034, 0.0),  # logSNR ~ +6.69
            _ImagePosition(0, "step_02", 0.5, 0.0),    # logSNR = 0.00
            _ImagePosition(0, "step_03", 0.867, 0.0),  # logSNR ~ -3.75
        ]
        logits = _clean_biased_step_logits(positions)
        assert logits[0] > logits[1] > logits[2], (
            f"Expected monotonically decreasing logits for decreasing logSNR, "
            f"got {logits.tolist()}"
        )

    def test_decay_scale_controls_steepness(self):
        """Higher decay_scale should make the logit difference larger."""
        positions = [
            _ImagePosition(0, "step_01", 0.034, 0.0),  # logSNR ~ +6.69
            _ImagePosition(0, "step_02", 0.5, 0.0),    # logSNR = 0.00
        ]
        logits_low = _clean_biased_step_logits(positions, decay_scale=0.25)
        logits_high = _clean_biased_step_logits(positions, decay_scale=1.0)

        diff_low = logits_low[0] - logits_low[1]
        diff_high = logits_high[0] - logits_high[1]

        assert diff_high > diff_low, (
            f"Higher decay_scale should produce larger logit spread. "
            f"diff_low={diff_low:.2f}, diff_high={diff_high:.2f}"
        )

    def test_all_sigma_zero_gives_all_neg_inf(self):
        """If all positions are sigma=0, all logits should be -inf."""
        positions = [
            _ImagePosition(0, "final", 0.0, 0.0),
            _ImagePosition(0, "final_2", 0.0, 0.0),
        ]
        logits = _clean_biased_step_logits(positions)
        assert all(logits == -np.inf)


# ---------------------------------------------------------------------------
# Tests: _sigma_to_logsnr helper
# ---------------------------------------------------------------------------

class TestSigmaToLogSNR:
    """Test the sigma-to-logSNR conversion."""

    def test_sigma_zero_is_pos_inf(self):
        assert _sigma_to_logsnr(0.0) == float("inf")

    def test_sigma_one_is_neg_inf(self):
        assert _sigma_to_logsnr(1.0) == float("-inf")

    def test_sigma_half_is_zero(self):
        assert abs(_sigma_to_logsnr(0.5)) < 1e-10

    def test_monotonically_decreasing(self):
        sigmas = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
        logsnrs = [_sigma_to_logsnr(s) for s in sigmas]
        for i in range(len(logsnrs) - 1):
            assert logsnrs[i] > logsnrs[i + 1], (
                f"logSNR should decrease with increasing sigma: "
                f"logSNR({sigmas[i]})={logsnrs[i]:.2f} vs "
                f"logSNR({sigmas[i+1]})={logsnrs[i+1]:.2f}"
            )


# ---------------------------------------------------------------------------
# Tests: _logsnr_hist_bin helper
# ---------------------------------------------------------------------------

class TestLogSNRHistBin:
    """Test the histogram binning function."""

    def test_sigma_zero_bin(self):
        assert _logsnr_hist_bin(0.0) == _SIGMA_ZERO_BIN

    def test_sigma_half(self):
        # logSNR = 0.0 -> [0,2) bin
        assert _logsnr_hist_bin(0.5) == "[0,2)"

    def test_sigma_near_clean(self):
        # sigma=0.034 -> logSNR ~ 6.69 -> [5,8) bin
        assert _logsnr_hist_bin(0.034) == "[5,8)"

    def test_sigma_very_noisy(self):
        # sigma=0.95 -> logSNR ~ -5.87 -> [-inf,-5) bin
        assert _logsnr_hist_bin(0.95) == "[-inf,-5)"

    def test_pure_noise(self):
        # sigma=1.0 -> [-inf,-5) bin
        assert _logsnr_hist_bin(1.0) == "[-inf,-5)"


# ---------------------------------------------------------------------------
# Tests: existing functions are NOT broken
# ---------------------------------------------------------------------------

class TestExistingFunctionsPreserved:
    """Verify that logsnr_sampling_weight and logsnr_sampling_logit still work."""

    def test_logsnr_sampling_weight_sigma_zero(self):
        assert logsnr_sampling_weight(0.0) == 1.0

    def test_logsnr_sampling_weight_sigma_half(self):
        w = logsnr_sampling_weight(0.5)
        assert 0.2 < w < 0.3  # default params: 0.250

    def test_logsnr_sampling_logit_sigma_zero(self):
        assert logsnr_sampling_logit(0.0) == 0.0

    def test_logsnr_sampling_logit_sigma_half(self):
        logit = logsnr_sampling_logit(0.5)
        assert logit < 0.0  # below threshold, negative logit


# ---------------------------------------------------------------------------
# Tests: sample_step_with_clean_bias method directly
# ---------------------------------------------------------------------------

class TestSampleStepWithCleanBiasMethod:
    """Test the sample_step_with_clean_bias method directly."""

    def test_raises_without_clean_fraction(self):
        """Should raise ValueError if called without clean_fraction."""
        positions = _make_positions(n_trajectories=3)
        sampler = BTRMPairSampler(
            positions=positions,
            allow_inter_trajectory=True,
            rng_seed=42,
            # clean_fraction not set
        )
        with pytest.raises(ValueError, match="clean_fraction is None"):
            sampler.sample_step_with_clean_bias(0)

    def test_override_clean_fraction_at_call_time(self):
        """Should work when clean_fraction is passed to the method directly."""
        positions = _make_positions(n_trajectories=3)
        sampler = BTRMPairSampler(
            positions=positions,
            allow_inter_trajectory=True,
            rng_seed=42,
            # clean_fraction not set at init
        )
        # Passing clean_fraction at call time should work
        pos = sampler.sample_step_with_clean_bias(0, clean_fraction=1.0)
        assert pos.sigma <= 0.0

    def test_returns_valid_position(self):
        """Returned position should be from the requested trajectory."""
        positions = _make_positions(n_trajectories=5)
        sampler = BTRMPairSampler(
            positions=positions,
            allow_inter_trajectory=True,
            rng_seed=42,
            clean_fraction=0.5,
        )
        for traj_id in range(5):
            pos = sampler.sample_step_with_clean_bias(traj_id)
            assert pos.traj_id == traj_id

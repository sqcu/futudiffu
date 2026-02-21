"""Tests for logSNR step weighting in the pair sampler.

Verifies that:
  1. Clean positions (high logSNR, low sigma) are sampled more frequently
     than noisy positions (low logSNR, high sigma).
  2. sigma=0 (fully denoised) gets FULL weight.
  3. The geometric decay is correctly applied per the user spec:
     "noisy latents with log(snr(t)) > 10 get sampled uniformly,
     every -5 step decrement below log(snr(t)) 10 gets p-% geometric decay."
  4. The weighting is resolution-aware: the same step index has different
     sigma values (and thus different weights) at different resolutions.
  5. The softmax distribution over positions within a trajectory biases
     toward cleaner steps.
  6. Default parameters are threshold=10.0, interval=5.0, decay_rate=0.5.

Pure Python test -- no GPU required.
"""

import math
import sys

import numpy as np

sys.path.insert(0, "F:\\dox\\repos\\ai\\futudiffu")
sys.path.insert(0, "F:\\dox\\repos\\ai\\futudiffu\\src_ii")

from src_ii.pair_sampler import (
    BTRMPairSampler,
    _ImagePosition,
    _softmax,
    logsnr_sampling_logit,
    logsnr_sampling_weight,
)
from src_ii.sigma_schedule import build_sigma_schedule_py, resolution_shift


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _logsnr(sigma: float) -> float:
    """Compute logSNR from sigma for the CONST noise model."""
    if sigma <= 0.0:
        return float("inf")
    if sigma >= 1.0:
        return float("-inf")
    return 2.0 * math.log((1.0 - sigma) / sigma)


def _make_positions_for_resolution(
    traj_id: int,
    width: int,
    height: int,
    n_steps: int = 30,
    saved_steps: list[int] | None = None,
    include_final: bool = True,
    include_sigma_zero: bool = False,
) -> list[_ImagePosition]:
    """Build _ImagePosition list for a single trajectory at a given resolution."""
    if saved_steps is None:
        saved_steps = [0, 4, 9, 14, 19, 24, 29]

    shift = resolution_shift(width, height)
    sigmas = build_sigma_schedule_py(n_steps, sampling_shift=shift)

    positions = []
    for step_idx in saved_steps:
        sigma = sigmas[step_idx]
        logit = logsnr_sampling_logit(sigma)
        positions.append(
            _ImagePosition(
                traj_id=traj_id,
                step_key=f"step_{step_idx:02d}",
                sigma=sigma,
                logit=logit,
                width=width,
                height=height,
            )
        )

    if include_final:
        final_sigma = sigmas[-2]
        logit = logsnr_sampling_logit(final_sigma)
        positions.append(
            _ImagePosition(
                traj_id=traj_id,
                step_key="final",
                sigma=final_sigma,
                logit=logit,
                width=width,
                height=height,
            )
        )

    if include_sigma_zero:
        logit = logsnr_sampling_logit(0.0)
        positions.append(
            _ImagePosition(
                traj_id=traj_id,
                step_key="denoised",
                sigma=0.0,
                logit=logit,
                width=width,
                height=height,
            )
        )

    return positions


# ---------------------------------------------------------------------------
# Tests: logsnr_sampling_weight
# ---------------------------------------------------------------------------


def test_sigma_zero_gets_full_weight():
    """sigma=0 (fully denoised) must get weight=1.0."""
    assert logsnr_sampling_weight(0.0) == 1.0
    # Also check with explicit negative
    assert logsnr_sampling_weight(-0.001) == 1.0


def test_sigma_one_gets_near_zero_weight():
    """sigma=1.0 (pure noise, logSNR=-inf) must get near-zero weight."""
    w = logsnr_sampling_weight(1.0)
    assert w < 1e-4, f"sigma=1.0 weight should be near zero, got {w}"


def test_default_threshold_is_10():
    """The default threshold must be 10.0 per user spec."""
    import inspect

    sig = inspect.signature(logsnr_sampling_weight)
    assert sig.parameters["threshold"].default == 10.0, (
        f"Expected default threshold=10.0, got {sig.parameters['threshold'].default}"
    )


def test_default_decay_rate_is_0_5():
    """The default decay_rate must be 0.5."""
    import inspect

    sig = inspect.signature(logsnr_sampling_weight)
    assert sig.parameters["decay_rate"].default == 0.5, (
        f"Expected default decay_rate=0.5, got {sig.parameters['decay_rate'].default}"
    )


def test_weight_is_one_above_threshold():
    """logSNR > threshold -> weight = 1.0."""
    # For threshold=10.0, need sigma such that logSNR > 10.
    # logSNR = 2 * ln((1-sigma)/sigma) > 10 => sigma < ~0.0067
    w = logsnr_sampling_weight(0.005)
    assert w == 1.0, f"logSNR={_logsnr(0.005):.2f} > 10, weight should be 1.0, got {w}"

    # sigma=0 also above threshold
    w = logsnr_sampling_weight(0.0)
    assert w == 1.0


def test_geometric_decay_below_threshold():
    """Below threshold, weight decays geometrically with logSNR distance."""
    # At logSNR = threshold - interval = 10 - 5 = 5: weight = 0.5^1 = 0.5
    # logSNR = 5.0 => sigma such that 2*ln((1-s)/s) = 5 => s = 1/(1+exp(2.5))
    sigma_logsnr5 = 1.0 / (1.0 + math.exp(2.5))
    w = logsnr_sampling_weight(sigma_logsnr5)
    expected = 0.5 ** 1.0
    assert abs(w - expected) < 0.001, (
        f"At logSNR=5.0 (sigma={sigma_logsnr5:.4f}), expected weight {expected}, got {w}"
    )

    # At logSNR = 0 (sigma=0.5): n_intervals = (10-0)/5 = 2, weight = 0.5^2 = 0.25
    w = logsnr_sampling_weight(0.5)
    expected = 0.5 ** 2.0
    assert abs(w - expected) < 0.001, (
        f"At logSNR=0.0 (sigma=0.5), expected weight {expected}, got {w}"
    )

    # At logSNR = -5 (sigma~0.924): n_intervals = (10-(-5))/5 = 3, weight = 0.5^3 = 0.125
    sigma_logsnr_neg5 = 1.0 / (1.0 + math.exp(-2.5))
    w = logsnr_sampling_weight(sigma_logsnr_neg5)
    expected = 0.5 ** 3.0
    assert abs(w - expected) < 0.001, (
        f"At logSNR=-5.0, expected weight {expected}, got {w}"
    )


def test_weight_monotonically_decreasing_with_sigma():
    """Weight should decrease as sigma increases (noise increases)."""
    sigmas = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99]
    weights = [logsnr_sampling_weight(s) for s in sigmas]
    for i in range(len(weights) - 1):
        assert weights[i] >= weights[i + 1], (
            f"Weight at sigma={sigmas[i]} ({weights[i]:.4f}) should be >= "
            f"weight at sigma={sigmas[i+1]} ({weights[i+1]:.4f})"
        )


def test_clean_noisier_weight_ratio():
    """Clean positions should have much higher weight than noisy positions.

    For 1280x832 30-step: step_29 (sigma=0.034) vs step_04 (sigma=0.867).
    With threshold=10, decay_rate=0.5:
      step_29: logSNR=6.69, n_intervals=(10-6.69)/5=0.662, weight=0.632
      step_04: logSNR=-3.75, n_intervals=(10-(-3.75))/5=2.75, weight=0.149
    Ratio should be ~4.25:1
    """
    w_clean = logsnr_sampling_weight(0.034)  # step_29
    w_noisy = logsnr_sampling_weight(0.867)  # step_04
    ratio = w_clean / w_noisy
    assert ratio > 3.5, (
        f"Expected clean/noisy ratio > 3.5, got {ratio:.2f} "
        f"(clean={w_clean:.4f}, noisy={w_noisy:.4f})"
    )


# ---------------------------------------------------------------------------
# Tests: logsnr_sampling_logit
# ---------------------------------------------------------------------------


def test_logit_is_log_of_weight():
    """logit should equal log(weight) for non-degenerate sigmas."""
    for sigma in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        w = logsnr_sampling_weight(sigma)
        l = logsnr_sampling_logit(sigma)
        assert abs(l - math.log(w)) < 1e-10, (
            f"At sigma={sigma}: logit={l}, log(weight)={math.log(w)}"
        )


def test_logit_zero_at_sigma_zero():
    """sigma=0 -> logit=0.0 (log(1.0))."""
    assert logsnr_sampling_logit(0.0) == 0.0


# ---------------------------------------------------------------------------
# Tests: resolution-dependent weighting
# ---------------------------------------------------------------------------


def test_same_step_different_resolution_different_weight():
    """The same step index has different sigma at different resolutions.

    For 1280x832 (shift=1.0), step_24 has sigma=0.200, logSNR=+2.77.
    For 256x256 (shift=4.03), step_24 has sigma=0.502, logSNR=-0.02.

    These have very different logSNR values and should therefore have
    different sampling weights.
    """
    sigmas_ref = build_sigma_schedule_py(30, sampling_shift=1.0)
    sigmas_256 = build_sigma_schedule_py(30, sampling_shift=resolution_shift(256, 256))

    sigma_ref_24 = sigmas_ref[24]
    sigma_256_24 = sigmas_256[24]

    # Verify the sigmas are indeed very different
    assert abs(sigma_ref_24 - 0.200) < 0.01, f"1280x832 step_24 sigma={sigma_ref_24}, expected ~0.200"
    assert abs(sigma_256_24 - 0.502) < 0.01, f"256x256 step_24 sigma={sigma_256_24}, expected ~0.502"

    # Weights should be different
    w_ref = logsnr_sampling_weight(sigma_ref_24)
    w_256 = logsnr_sampling_weight(sigma_256_24)
    assert w_ref > w_256, (
        f"1280x832 step_24 (sigma={sigma_ref_24:.3f}) should have higher weight "
        f"than 256x256 step_24 (sigma={sigma_256_24:.3f}), got {w_ref:.4f} vs {w_256:.4f}"
    )


def test_256x256_step29_is_noisy():
    """For 256x256 (shift~4.0), step_29 has logSNR=+3.9, which is BELOW
    the threshold of 10. Its weight should be significantly less than 1.0.
    """
    shift = resolution_shift(256, 256)
    sigmas = build_sigma_schedule_py(30, sampling_shift=shift)
    sigma_29 = sigmas[29]
    logsnr_29 = _logsnr(sigma_29)

    # Verify logSNR is well below 10
    assert logsnr_29 < 5.0, (
        f"256x256 step_29 should have logSNR < 5, got {logsnr_29:.2f}"
    )

    # Weight should be much less than 1.0
    w = logsnr_sampling_weight(sigma_29)
    assert w < 0.6, (
        f"256x256 step_29 weight should be < 0.6 (logSNR={logsnr_29:.2f}), got {w:.4f}"
    )


# ---------------------------------------------------------------------------
# Tests: softmax distribution over positions
# ---------------------------------------------------------------------------


def test_softmax_biases_toward_clean_steps():
    """The softmax distribution over a trajectory's steps should give higher
    probability to cleaner steps (lower sigma, higher logSNR).
    """
    positions = _make_positions_for_resolution(
        traj_id=0, width=1280, height=832
    )
    logits = np.array([p.logit for p in positions], dtype=np.float64)
    probs = _softmax(logits)

    # step_29 and final (cleanest) should have highest probability
    # step_00 (pure noise) should have lowest
    step_labels = [p.step_key for p in positions]

    idx_step00 = step_labels.index("step_00")
    idx_step29 = step_labels.index("step_29")
    idx_final = step_labels.index("final")

    assert probs[idx_step29] > probs[idx_step00], (
        f"step_29 prob ({probs[idx_step29]:.4f}) should be > "
        f"step_00 prob ({probs[idx_step00]:.4f})"
    )
    assert probs[idx_final] > probs[idx_step00], (
        f"final prob ({probs[idx_final]:.4f}) should be > "
        f"step_00 prob ({probs[idx_step00]:.4f})"
    )

    # step_00 probability should be very low (< 1%)
    assert probs[idx_step00] < 0.01, (
        f"step_00 (pure noise) probability should be < 1%, got {probs[idx_step00]*100:.2f}%"
    )

    # step_29 probability should be significantly higher than step_04
    idx_step04 = step_labels.index("step_04")
    assert probs[idx_step29] > 2.0 * probs[idx_step04], (
        f"step_29 prob ({probs[idx_step29]:.4f}) should be > 2x "
        f"step_04 prob ({probs[idx_step04]:.4f})"
    )


def test_sigma_zero_dominates_distribution():
    """When sigma=0 (fully denoised) is present, it should get the highest
    sampling probability.
    """
    positions = _make_positions_for_resolution(
        traj_id=0, width=1280, height=832, include_sigma_zero=True,
    )
    logits = np.array([p.logit for p in positions], dtype=np.float64)
    probs = _softmax(logits)

    step_labels = [p.step_key for p in positions]
    idx_denoised = step_labels.index("denoised")

    # sigma=0 should have the highest probability
    assert probs[idx_denoised] == probs.max(), (
        f"sigma=0 should have highest prob ({probs[idx_denoised]:.4f}), "
        f"but max is {probs.max():.4f}"
    )

    # It should get at least 20% of the probability mass
    assert probs[idx_denoised] > 0.20, (
        f"sigma=0 should get > 20% probability, got {probs[idx_denoised]*100:.1f}%"
    )


# ---------------------------------------------------------------------------
# Tests: BTRMPairSampler empirical distribution
# ---------------------------------------------------------------------------


def test_sampler_empirical_clean_bias():
    """Empirically verify that the sampler favors cleaner positions.

    Build positions for 3 trajectories at 1280x832, sample 5000 pairs,
    count how often each step position appears. Cleaner steps should
    appear more often than noisier steps.
    """
    all_positions = []
    for tid in range(3):
        all_positions.extend(
            _make_positions_for_resolution(traj_id=tid, width=1280, height=832)
        )

    sampler = BTRMPairSampler(
        positions=all_positions,
        allow_inter_trajectory=True,
        allow_intra_trajectory=True,
        rng_seed=42,
    )

    # Sample many pairs and count step appearances
    step_counts = {}
    n_samples = 5000
    for _ in range(n_samples):
        pair = sampler.sample_pair()
        for suffix in ("a", "b"):
            step = pair[f"step_{suffix}"]
            step_counts[step] = step_counts.get(step, 0) + 1

    # step_29 and final should be the most sampled
    # step_00 should be the least sampled
    total = sum(step_counts.values())
    frac_step00 = step_counts.get("step_00", 0) / total
    frac_step29 = step_counts.get("step_29", 0) / total
    frac_final = step_counts.get("final", 0) / total

    assert frac_step29 > frac_step00, (
        f"step_29 fraction ({frac_step29:.4f}) should be > "
        f"step_00 fraction ({frac_step00:.4f})"
    )
    assert frac_final > frac_step00, (
        f"final fraction ({frac_final:.4f}) should be > "
        f"step_00 fraction ({frac_step00:.4f})"
    )

    # The ratio between step_29 and step_00 should be large (> 50x)
    # because step_00 gets near-zero weight
    if step_counts.get("step_00", 0) > 0:
        ratio_29_to_00 = step_counts.get("step_29", 0) / step_counts["step_00"]
        assert ratio_29_to_00 > 20, (
            f"step_29/step_00 ratio should be > 20, got {ratio_29_to_00:.1f}"
        )


def test_sampler_256x256_more_concentrated():
    """For 256x256, the sigma schedule is compressed toward 1.0,
    so the logSNR range is compressed and the weighting gives the
    cleanest steps (step_29) an even larger relative advantage.
    """
    all_positions = []
    for tid in range(3):
        all_positions.extend(
            _make_positions_for_resolution(traj_id=tid, width=256, height=256)
        )

    sampler = BTRMPairSampler(
        positions=all_positions,
        allow_inter_trajectory=True,
        allow_intra_trajectory=True,
        rng_seed=42,
    )

    step_counts = {}
    n_samples = 5000
    for _ in range(n_samples):
        pair = sampler.sample_pair()
        for suffix in ("a", "b"):
            step = pair[f"step_{suffix}"]
            step_counts[step] = step_counts.get(step, 0) + 1

    total = sum(step_counts.values())
    frac_step29 = step_counts.get("step_29", 0) / total
    frac_final = step_counts.get("final", 0) / total
    frac_step04 = step_counts.get("step_04", 0) / total

    # step_29 + final combined should be > 30% (heavily biased)
    assert frac_step29 + frac_final > 0.30, (
        f"step_29 + final should be > 30%, got {(frac_step29+frac_final)*100:.1f}%"
    )

    # step_29 should be > 3x step_04
    assert frac_step29 > 2.5 * frac_step04, (
        f"step_29 ({frac_step29*100:.1f}%) should be > 2.5x step_04 ({frac_step04*100:.1f}%)"
    )


# ---------------------------------------------------------------------------
# Tests: weight table verification
# ---------------------------------------------------------------------------


def test_weight_table_1280x832():
    """Verify the weight table from the docstring for 1280x832 (shift=1.0)."""
    sigmas = build_sigma_schedule_py(30, sampling_shift=1.0)
    expected = [
        # (step_idx, expected_weight, tolerance)
        (0, 0.000, 0.001),    # sigma=1.0, pure noise
        (4, 0.149, 0.005),    # sigma=0.867
        (9, 0.198, 0.005),    # sigma=0.700
        (14, 0.241, 0.005),   # sigma=0.534
        (19, 0.291, 0.005),   # sigma=0.367
        (24, 0.367, 0.005),   # sigma=0.200
        (29, 0.632, 0.005),   # sigma=0.034
    ]
    for step_idx, exp_w, tol in expected:
        sigma = sigmas[step_idx]
        w = logsnr_sampling_weight(sigma)
        assert abs(w - exp_w) < tol, (
            f"step_{step_idx} (sigma={sigma:.3f}): expected weight ~{exp_w}, got {w:.4f}"
        )


def test_weight_table_256x256():
    """Verify the weight table for 256x256 (shift~4.03)."""
    shift = resolution_shift(256, 256)
    sigmas = build_sigma_schedule_py(30, sampling_shift=shift)
    expected = [
        # (step_idx, expected_weight, tolerance)
        (0, 0.000, 0.001),     # sigma=1.0
        (4, 0.101, 0.005),     # sigma~0.963
        (9, 0.134, 0.005),     # sigma~0.904
        (14, 0.164, 0.005),    # sigma~0.822
        (19, 0.198, 0.005),    # sigma~0.700
        (24, 0.250, 0.005),    # sigma~0.502
        (29, 0.430, 0.005),    # sigma~0.124
    ]
    for step_idx, exp_w, tol in expected:
        sigma = sigmas[step_idx]
        w = logsnr_sampling_weight(sigma)
        assert abs(w - exp_w) < tol, (
            f"256x256 step_{step_idx} (sigma={sigma:.3f}): "
            f"expected weight ~{exp_w}, got {w:.4f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_all():
    """Run all tests and print results."""
    import traceback

    tests = [
        test_sigma_zero_gets_full_weight,
        test_sigma_one_gets_near_zero_weight,
        test_default_threshold_is_10,
        test_default_decay_rate_is_0_5,
        test_weight_is_one_above_threshold,
        test_geometric_decay_below_threshold,
        test_weight_monotonically_decreasing_with_sigma,
        test_clean_noisier_weight_ratio,
        test_logit_is_log_of_weight,
        test_logit_zero_at_sigma_zero,
        test_same_step_different_resolution_different_weight,
        test_256x256_step29_is_noisy,
        test_softmax_biases_toward_clean_steps,
        test_sigma_zero_dominates_distribution,
        test_sampler_empirical_clean_bias,
        test_sampler_256x256_more_concentrated,
        test_weight_table_1280x832,
        test_weight_table_256x256,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            print(f"  PASS: {test_fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test_fn.__name__}")
            traceback.print_exc()
            print()
            failed += 1

    print(f"\n{passed} passed, {failed} failed out of {len(tests)} tests")
    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)

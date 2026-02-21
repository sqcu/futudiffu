"""Tests for compute_logsnr_uniform_steps in src_ii/sigma_schedule.py.

Validates that logSNR-uniform step selection produces better logSNR spacing
than the default uniform-index SPARSE_STEPS = {0, 4, 9, 14, 19, 24, 29}.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src_ii.sigma_schedule import (
    build_sigma_schedule,
    compute_logsnr_uniform_steps,
    resolution_shift,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SIGMA_MIN = 0.001
SIGMA_MAX = 0.999

DEFAULT_SPARSE_STEPS = [0, 4, 9, 14, 19, 24, 29]


def _logsnr_at_step(sigmas, step_idx: int) -> float:
    """Compute logSNR for a given step index, clamping sigma."""
    s = float(sigmas[step_idx].item())
    s = max(SIGMA_MIN, min(SIGMA_MAX, s))
    return 2.0 * math.log((1.0 - s) / s)


def _logsnr_spacing_stddev(logsnr_values: list[float]) -> float:
    """Standard deviation of consecutive logSNR gaps.

    A perfectly uniform spacing would have stddev=0.
    Lower is better (more uniform).
    """
    gaps = [logsnr_values[i + 1] - logsnr_values[i] for i in range(len(logsnr_values) - 1)]
    if len(gaps) < 2:
        return 0.0
    mean_gap = sum(gaps) / len(gaps)
    variance = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
    return math.sqrt(variance)


def _get_logsnrs_for_steps(
    width: int, height: int, steps: list[int], n_steps: int = 30,
) -> list[float]:
    """Get logSNR values for a list of step indices at a given resolution."""
    import torch

    shift = resolution_shift(width, height)
    sigmas = build_sigma_schedule(
        n_steps, sampling_shift=shift, device=torch.device("cpu"), dtype=torch.float32,
    )
    return [_logsnr_at_step(sigmas, i) for i in steps]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBasicProperties:
    """Test structural properties of compute_logsnr_uniform_steps."""

    def test_always_includes_step_0_and_last(self):
        """Step 0 and step n_steps-1 must always be in the result."""
        for w, h in [(1280, 832), (256, 256), (512, 512), (1408, 736)]:
            steps = compute_logsnr_uniform_steps(w, h, n_steps=30, n_save=7)
            assert 0 in steps, f"Step 0 missing for {w}x{h}"
            assert 29 in steps, f"Step 29 missing for {w}x{h}"

    def test_sorted_and_correct_length(self):
        """Result must be sorted and have exactly n_save elements."""
        for n_save in [3, 5, 7, 10, 15]:
            steps = compute_logsnr_uniform_steps(1280, 832, n_steps=30, n_save=n_save)
            assert len(steps) == n_save, f"Expected {n_save} steps, got {len(steps)}"
            assert steps == sorted(steps), f"Steps not sorted: {steps}"

    def test_all_indices_in_range(self):
        """All step indices must be in [0, n_steps-1]."""
        steps = compute_logsnr_uniform_steps(1280, 832, n_steps=30, n_save=7)
        for s in steps:
            assert 0 <= s <= 29, f"Step {s} out of range [0, 29]"

    def test_no_duplicates(self):
        """All step indices must be unique."""
        steps = compute_logsnr_uniform_steps(1280, 832, n_steps=30, n_save=7)
        assert len(steps) == len(set(steps)), f"Duplicate steps: {steps}"

    def test_n_save_equals_2(self):
        """Edge case: n_save=2 should return [0, n_steps-1]."""
        steps = compute_logsnr_uniform_steps(1280, 832, n_steps=30, n_save=2)
        assert steps == [0, 29]

    def test_n_save_equals_n_steps(self):
        """Edge case: n_save=n_steps should return all step indices."""
        steps = compute_logsnr_uniform_steps(1280, 832, n_steps=10, n_save=10)
        assert steps == list(range(10))

    def test_invalid_n_save(self):
        """n_save < 2 or n_save > n_steps should raise ValueError."""
        with pytest.raises(ValueError):
            compute_logsnr_uniform_steps(1280, 832, n_steps=30, n_save=1)
        with pytest.raises(ValueError):
            compute_logsnr_uniform_steps(1280, 832, n_steps=30, n_save=31)


class TestLogSNRUniformity:
    """Test that logSNR-uniform steps are more evenly spaced than default."""

    def test_more_uniform_than_default_1280x832(self):
        """At 1280x832 (shift=1.0), logSNR-uniform steps should have smaller
        spacing stddev than the default uniform-index steps."""
        w, h = 1280, 832
        uniform_steps = compute_logsnr_uniform_steps(w, h, n_steps=30, n_save=7)
        default_steps = DEFAULT_SPARSE_STEPS

        logsnrs_uniform = _get_logsnrs_for_steps(w, h, uniform_steps)
        logsnrs_default = _get_logsnrs_for_steps(w, h, default_steps)

        std_uniform = _logsnr_spacing_stddev(logsnrs_uniform)
        std_default = _logsnr_spacing_stddev(logsnrs_default)

        print(f"\n1280x832 (shift=1.0):")
        print(f"  Default steps:  {default_steps}")
        print(f"  Default logSNR: {['%.2f' % v for v in logsnrs_default]}")
        print(f"  Default stddev: {std_default:.4f}")
        print(f"  Uniform steps:  {uniform_steps}")
        print(f"  Uniform logSNR: {['%.2f' % v for v in logsnrs_uniform]}")
        print(f"  Uniform stddev: {std_uniform:.4f}")
        print(f"  Improvement:    {std_default / max(std_uniform, 1e-10):.2f}x")

        assert std_uniform < std_default, (
            f"logSNR-uniform steps (stddev={std_uniform:.4f}) should be more uniform "
            f"than default (stddev={std_default:.4f}) at 1280x832"
        )

    def test_more_uniform_than_default_256x256(self):
        """At 256x256 (high shift), the improvement should be even more pronounced
        because the sigma schedule is heavily compressed."""
        w, h = 256, 256
        uniform_steps = compute_logsnr_uniform_steps(w, h, n_steps=30, n_save=7)
        default_steps = DEFAULT_SPARSE_STEPS

        logsnrs_uniform = _get_logsnrs_for_steps(w, h, uniform_steps)
        logsnrs_default = _get_logsnrs_for_steps(w, h, default_steps)

        std_uniform = _logsnr_spacing_stddev(logsnrs_uniform)
        std_default = _logsnr_spacing_stddev(logsnrs_default)

        print(f"\n256x256 (high shift):")
        print(f"  Default steps:  {default_steps}")
        print(f"  Default logSNR: {['%.2f' % v for v in logsnrs_default]}")
        print(f"  Default stddev: {std_default:.4f}")
        print(f"  Uniform steps:  {uniform_steps}")
        print(f"  Uniform logSNR: {['%.2f' % v for v in logsnrs_uniform]}")
        print(f"  Uniform stddev: {std_uniform:.4f}")
        print(f"  Improvement:    {std_default / max(std_uniform, 1e-10):.2f}x")

        assert std_uniform < std_default, (
            f"logSNR-uniform steps (stddev={std_uniform:.4f}) should be more uniform "
            f"than default (stddev={std_default:.4f}) at 256x256"
        )


class TestResolutionDependence:
    """Test that different resolutions produce different step selections."""

    def test_different_resolutions_different_steps(self):
        """256x256 (shift~4.0) and 1280x832 (shift=1.0) should produce
        different step selections because the sigma schedule differs."""
        steps_small = compute_logsnr_uniform_steps(256, 256, n_steps=30, n_save=7)
        steps_large = compute_logsnr_uniform_steps(1280, 832, n_steps=30, n_save=7)

        print(f"\n256x256 steps:  {steps_small}")
        print(f"1280x832 steps: {steps_large}")

        assert steps_small != steps_large, (
            f"Expected different steps for 256x256 vs 1280x832, "
            f"but both returned {steps_small}"
        )

    def test_high_shift_more_steps_at_end(self):
        """For high-shift resolutions (small images), sigma is compressed toward 1.0,
        so most of the logSNR range is in the later steps. The logSNR-uniform
        selection should place more steps in the second half (indices >= 15)."""
        steps_small = compute_logsnr_uniform_steps(256, 256, n_steps=30, n_save=7)
        steps_large = compute_logsnr_uniform_steps(1280, 832, n_steps=30, n_save=7)

        # Count steps in the second half (indices 15-29)
        second_half_small = sum(1 for s in steps_small if s >= 15)
        second_half_large = sum(1 for s in steps_large if s >= 15)

        print(f"\n256x256 steps in [15,29]:  {second_half_small} of {steps_small}")
        print(f"1280x832 steps in [15,29]: {second_half_large} of {steps_large}")

        assert second_half_small >= second_half_large, (
            f"Small images (high shift) should have at least as many steps in the "
            f"second half as large images. Got small={second_half_small}, "
            f"large={second_half_large}"
        )


class TestMultipleResolutions:
    """Test across the full resolution range to verify no crashes or degenerate outputs."""

    @pytest.mark.parametrize(
        "width,height",
        [
            (224, 224),
            (256, 256),
            (320, 320),
            (384, 384),
            (512, 512),
            (704, 704),
            (768, 512),
            (1024, 1024),
            (1280, 832),
            (1408, 736),
            (832, 1280),  # portrait
        ],
    )
    def test_resolution_produces_valid_output(self, width, height):
        """Every resolution from 224x224 to 1408x736 should produce valid output."""
        steps = compute_logsnr_uniform_steps(width, height, n_steps=30, n_save=7)
        assert len(steps) == 7
        assert steps == sorted(steps)
        assert 0 in steps
        assert 29 in steps
        assert len(set(steps)) == 7


class TestVisualInspection:
    """Print detailed logSNR tables for visual inspection."""

    def test_print_all_resolutions(self):
        """Print selected steps and logSNR values for several resolutions."""
        import torch

        resolutions = [
            (256, 256, "tiny"),
            (512, 512, "small"),
            (768, 512, "medium landscape"),
            (1024, 1024, "megapixel"),
            (1280, 832, "reference"),
            (1408, 736, "wide"),
        ]

        print("\n" + "=" * 80)
        print("LogSNR-Uniform Step Selection Across Resolutions")
        print("=" * 80)

        for w, h, label in resolutions:
            shift = resolution_shift(w, h)
            sigmas = build_sigma_schedule(
                30, sampling_shift=shift,
                device=torch.device("cpu"), dtype=torch.float32,
            )

            uniform_steps = compute_logsnr_uniform_steps(w, h, n_steps=30, n_save=7)

            print(f"\n{w}x{h} ({label}, shift={shift:.3f}):")
            print(f"  Selected steps: {uniform_steps}")
            print(f"  Default steps:  {DEFAULT_SPARSE_STEPS}")

            print(f"  {'step':>6} {'sigma':>8} {'logSNR':>8}  selected?")
            for i in range(30):
                s = float(sigmas[i].item())
                sc = max(SIGMA_MIN, min(SIGMA_MAX, s))
                lsnr = 2.0 * math.log((1.0 - sc) / sc)
                marker = " <--" if i in uniform_steps else ""
                default_marker = " [D]" if i in DEFAULT_SPARSE_STEPS else ""
                print(f"  {i:>6} {s:>8.4f} {lsnr:>8.2f}{marker}{default_marker}")

        # This test always passes -- it's for visual inspection
        assert True

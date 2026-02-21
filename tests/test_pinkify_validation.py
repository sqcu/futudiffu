"""Tests for src_ii/pinkify_validation.py.

Tests the ranking validation logic with both mock scores and real GPU scoring.
GPU tests are skipped if CUDA is unavailable.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src_ii.pinkify_validation import (
    CHALLENGE_LABELS,
    _check_ranking,
    _load_challenge_images,
    validate_pinkify_ranking,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CHALLENGE_DIR = REPO_ROOT / "i2i_off_policies" / "PINKIFY_cases"
_CHALLENGE_DIR_EXISTS = CHALLENGE_DIR.exists() and all(
    (CHALLENGE_DIR / f"PINKER_{label}.png").exists() for label in CHALLENGE_LABELS
)

_HAS_CUDA = False
try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except ImportError:
    pass

requires_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="CUDA not available")
requires_challenge_images = pytest.mark.skipif(
    not _CHALLENGE_DIR_EXISTS, reason="PINKIFY challenge images not on disk"
)


# ---------------------------------------------------------------------------
# Unit tests: _check_ranking with synthetic scores
# ---------------------------------------------------------------------------

class TestCheckRanking:
    """Test the ranking constraint logic with known scores."""

    def test_perfect_ranking(self):
        """All constraints satisfied with well-separated scores."""
        scores = {"A": 0.01, "B": 0.05, "C": 0.10, "D": 0.30, "E": 0.35, "F": 0.60}
        checks = _check_ranking(scores)
        assert all(c["passed"] for c in checks), (
            f"Expected all checks to pass: {checks}"
        )

    def test_a_not_less_than_b(self):
        """Fails when A >= B."""
        scores = {"A": 0.10, "B": 0.05, "C": 0.20, "D": 0.30, "E": 0.35, "F": 0.60}
        checks = _check_ranking(scores)
        a_lt_b = next(c for c in checks if c["name"] == "A < B")
        assert not a_lt_b["passed"]

    def test_b_not_less_than_c(self):
        """Fails when B >= C."""
        scores = {"A": 0.01, "B": 0.20, "C": 0.10, "D": 0.30, "E": 0.35, "F": 0.60}
        checks = _check_ranking(scores)
        b_lt_c = next(c for c in checks if c["name"] == "B < C")
        assert not b_lt_c["passed"]

    def test_abc_not_less_than_de(self):
        """Fails when max(A,B,C) >= min(D,E)."""
        scores = {"A": 0.01, "B": 0.05, "C": 0.40, "D": 0.30, "E": 0.35, "F": 0.60}
        checks = _check_ranking(scores)
        gap = next(c for c in checks if "max(A,B,C)" in c["name"])
        assert not gap["passed"]

    def test_d_e_equal(self):
        """D == E should pass the ~50% relative diff check."""
        scores = {"A": 0.01, "B": 0.05, "C": 0.10, "D": 0.30, "E": 0.30, "F": 0.60}
        checks = _check_ranking(scores)
        d_e_check = next(c for c in checks if "D ~ E" in c["name"])
        assert d_e_check["passed"], f"D == E should pass: {d_e_check}"

    def test_d_e_close_but_different(self):
        """D and E within 49% relative diff should pass."""
        scores = {"A": 0.01, "B": 0.05, "C": 0.10, "D": 0.30, "E": 0.44, "F": 0.60}
        checks = _check_ranking(scores)
        d_e_check = next(c for c in checks if "D ~ E" in c["name"])
        # |0.30 - 0.44| / max(0.30, 0.44) = 0.14 / 0.44 = 0.318 < 0.5
        assert d_e_check["passed"]

    def test_d_e_too_far_apart(self):
        """D and E with >50% relative diff should fail."""
        scores = {"A": 0.01, "B": 0.05, "C": 0.10, "D": 0.10, "E": 0.30, "F": 0.60}
        checks = _check_ranking(scores)
        d_e_check = next(c for c in checks if "D ~ E" in c["name"])
        # |0.10 - 0.30| / max(0.10, 0.30) = 0.20 / 0.30 = 0.667 > 0.5
        assert not d_e_check["passed"]

    def test_de_not_less_than_f(self):
        """Fails when max(D,E) >= F."""
        scores = {"A": 0.01, "B": 0.05, "C": 0.10, "D": 0.30, "E": 0.35, "F": 0.20}
        checks = _check_ranking(scores)
        de_lt_f = next(c for c in checks if "max(D,E) < F" in c["name"])
        assert not de_lt_f["passed"]

    def test_check_count(self):
        """Should always return exactly 5 checks."""
        scores = {"A": 0.01, "B": 0.05, "C": 0.10, "D": 0.30, "E": 0.35, "F": 0.60}
        checks = _check_ranking(scores)
        assert len(checks) == 5

    def test_check_fields(self):
        """Every check dict should have name, passed, detail."""
        scores = {"A": 0.01, "B": 0.05, "C": 0.10, "D": 0.30, "E": 0.35, "F": 0.60}
        checks = _check_ranking(scores)
        for c in checks:
            assert "name" in c
            assert "passed" in c
            assert "detail" in c
            assert isinstance(c["name"], str)
            assert isinstance(c["passed"], bool)
            assert isinstance(c["detail"], str)


# ---------------------------------------------------------------------------
# Unit tests: validate_pinkify_ranking with mock score_fn
# ---------------------------------------------------------------------------

class TestValidatePinkifyRankingMock:
    """Test validate_pinkify_ranking with a mock score function."""

    def _make_mock_score_fn(self, score_map: dict[str, float]):
        """Create a mock score_fn that returns scores based on image shape.

        Since we cannot match by label directly (score_fn only sees tensors),
        we use the image tensor's mean value as a proxy. This requires images
        to have distinct mean pixel values, which the PINKER challenge images
        do in practice.

        For mock testing, we bypass image loading entirely and test the return
        format with a known-good score function.
        """
        # We just return a fixed incrementing score per call
        call_count = [0]
        ordered_scores = [score_map[label] for label in CHALLENGE_LABELS]

        def mock_fn(tensor):
            import torch
            idx = call_count[0]
            call_count[0] += 1
            return torch.tensor(ordered_scores[idx])

        return mock_fn

    @requires_challenge_images
    def test_return_format_passing(self):
        """Validate return dict has all expected keys when ranking passes."""
        import torch

        # Perfect ranking: A < B < C < D ~ E < F
        scores = {"A": 0.01, "B": 0.05, "C": 0.10, "D": 0.30, "E": 0.35, "F": 0.60}
        mock_fn = self._make_mock_score_fn(scores)

        result = validate_pinkify_ranking(
            score_fn=mock_fn,
            challenge_dir=str(CHALLENGE_DIR),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        assert "passed" in result
        assert "scores" in result
        assert "checks" in result
        assert "rank_order" in result

        assert result["passed"] is True
        assert isinstance(result["scores"], dict)
        assert set(result["scores"].keys()) == set(CHALLENGE_LABELS)
        assert isinstance(result["checks"], list)
        assert len(result["checks"]) == 5
        assert isinstance(result["rank_order"], list)
        assert len(result["rank_order"]) == 6

    @requires_challenge_images
    def test_return_format_failing(self):
        """Validate return dict when ranking fails."""
        import torch

        # Inverted ranking: A > B (should fail A < B check)
        scores = {"A": 0.50, "B": 0.05, "C": 0.10, "D": 0.30, "E": 0.35, "F": 0.60}
        mock_fn = self._make_mock_score_fn(scores)

        result = validate_pinkify_ranking(
            score_fn=mock_fn,
            challenge_dir=str(CHALLENGE_DIR),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        assert result["passed"] is False
        # At least one check should have failed
        assert any(not c["passed"] for c in result["checks"])

    @requires_challenge_images
    def test_rank_order_is_sorted(self):
        """rank_order should be labels sorted by ascending score."""
        import torch

        scores = {"A": 0.01, "B": 0.05, "C": 0.10, "D": 0.30, "E": 0.35, "F": 0.60}
        mock_fn = self._make_mock_score_fn(scores)

        result = validate_pinkify_ranking(
            score_fn=mock_fn,
            challenge_dir=str(CHALLENGE_DIR),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        # Check that scores are monotonically increasing along rank_order
        ordered_scores = [result["scores"][label] for label in result["rank_order"]]
        for i in range(len(ordered_scores) - 1):
            assert ordered_scores[i] <= ordered_scores[i + 1], (
                f"rank_order not sorted: {result['rank_order']} with scores {ordered_scores}"
            )

    def test_missing_challenge_dir_raises(self):
        """FileNotFoundError when challenge directory does not exist."""
        import torch

        with pytest.raises(FileNotFoundError):
            validate_pinkify_ranking(
                score_fn=lambda t: torch.tensor(0.0),
                challenge_dir="/nonexistent/path/PINKIFY_cases",
                device=torch.device("cpu"),
            )


# ---------------------------------------------------------------------------
# GPU integration tests: real images + real pinkify_score_gpu
# ---------------------------------------------------------------------------

@requires_cuda
@requires_challenge_images
class TestValidatePinkifyRankingGPU:
    """Test validate_pinkify_ranking with real images and GPU scoring."""

    def test_real_pinkify_ranking_passes(self):
        """The real PINKIFY challenge images should satisfy all ranking constraints."""
        result = validate_pinkify_ranking(
            challenge_dir=str(CHALLENGE_DIR),
            device=torch.device("cuda"),
        )

        print(f"\nPINKIFY challenge set scores:")
        for label in CHALLENGE_LABELS:
            print(f"  {label}: {result['scores'][label]:.6f}")
        print(f"Rank order: {result['rank_order']}")
        for c in result["checks"]:
            status = "PASS" if c["passed"] else "FAIL"
            print(f"  [{status}] {c['name']}: {c['detail']}")

        assert result["passed"], (
            f"Expected all ranking constraints to hold. "
            f"Failed checks: {[c for c in result['checks'] if not c['passed']]}"
        )

    def test_scores_are_positive(self):
        """All PINKIFY scores should be non-negative (images have some pink)."""
        result = validate_pinkify_ranking(
            challenge_dir=str(CHALLENGE_DIR),
            device=torch.device("cuda"),
        )

        for label, score in result["scores"].items():
            assert score >= 0.0, f"Score for {label} is negative: {score}"

    def test_f_is_highest(self):
        """PINKER_F should have the highest score (it is the pinkest image)."""
        result = validate_pinkify_ranking(
            challenge_dir=str(CHALLENGE_DIR),
            device=torch.device("cuda"),
        )

        f_score = result["scores"]["F"]
        for label in ("A", "B", "C", "D", "E"):
            assert f_score > result["scores"][label], (
                f"F ({f_score:.6f}) should be > {label} ({result['scores'][label]:.6f})"
            )

    def test_rank_order_has_all_labels(self):
        """rank_order should contain all 6 labels."""
        result = validate_pinkify_ranking(
            challenge_dir=str(CHALLENGE_DIR),
            device=torch.device("cuda"),
        )

        assert set(result["rank_order"]) == set(CHALLENGE_LABELS)


# ---------------------------------------------------------------------------
# GPU integration tests: image loading
# ---------------------------------------------------------------------------

@requires_cuda
@requires_challenge_images
class TestLoadChallengeImages:
    """Test _load_challenge_images with real files."""

    def test_loads_all_six(self):
        """Should load all 6 images."""
        images = _load_challenge_images(
            str(CHALLENGE_DIR), device=torch.device("cuda"),
        )
        assert set(images.keys()) == set(CHALLENGE_LABELS)

    def test_shape_and_dtype(self):
        """Each image should be [3, H, W] float32."""
        images = _load_challenge_images(
            str(CHALLENGE_DIR), device=torch.device("cuda"),
        )
        for label, t in images.items():
            assert t.ndim == 3, f"{label}: expected 3 dims, got {t.ndim}"
            assert t.shape[0] == 3, f"{label}: expected 3 channels, got {t.shape[0]}"
            assert t.dtype == torch.float32, f"{label}: expected float32, got {t.dtype}"

    def test_value_range(self):
        """Pixel values should be in [0, 1]."""
        images = _load_challenge_images(
            str(CHALLENGE_DIR), device=torch.device("cuda"),
        )
        for label, t in images.items():
            assert t.min() >= 0.0, f"{label}: min={t.min():.4f}"
            assert t.max() <= 1.0, f"{label}: max={t.max():.4f}"

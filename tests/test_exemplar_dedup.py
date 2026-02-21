"""Test exemplar renderer deduplication across heads.

Verifies that when two heads produce correlated rankings, the
deduplicate_across_heads=True option (default) ensures each head
shows different exemplar images.

Pure Python test -- no GPU, no torch, no VAE. Tests the selection
logic only by mocking the VAE decode path.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))


def _make_fake_trajectories(n: int) -> list[dict]:
    """Create fake trajectory entries (no real latents)."""
    trajs = []
    for i in range(n):
        trajs.append({
            "traj_id": i,
            "step_key": "final",
            "latent": f"fake_latent_{i}",  # placeholder
        })
    return trajs


def _make_correlated_scores(n: int) -> dict[str, dict[str, float]]:
    """Create scores where both heads rank images in the same order.

    head_a scores: 1.0, 2.0, 3.0, ...
    head_b scores: 10.0, 20.0, 30.0, ...  (same ranking, different values)
    """
    scores = {}
    for i in range(n):
        key = f"{i}_final"
        scores[key] = {
            "head_a": float(i + 1),       # 1.0, 2.0, ..., n
            "head_b": float((i + 1) * 10),  # 10.0, 20.0, ..., n*10
        }
    return scores


def _make_anticorrelated_scores(n: int) -> dict[str, dict[str, float]]:
    """Create scores where heads rank images in opposite order.

    head_a scores: 1.0, 2.0, 3.0, ...
    head_b scores: n*10, (n-1)*10, ...  (reversed ranking)
    """
    scores = {}
    for i in range(n):
        key = f"{i}_final"
        scores[key] = {
            "head_a": float(i + 1),
            "head_b": float((n - i) * 10),
        }
    return scores


def test_correlated_heads_without_dedup():
    """Without deduplication, correlated heads pick the same images."""
    from src_ii.exemplar_renderer import render_exemplars

    n = 10
    trajs = _make_fake_trajectories(n)
    scores = _make_correlated_scores(n)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock VAE-related imports and the rendering function
        with patch("src_ii.exemplar_renderer._render_one") as mock_render:
            mock_render.side_effect = lambda out_dir, traj_lookup, img_key, vae, device, dtype, head, rank, score, category: {
                "filename": f"{head}_{category}{rank:02d}.png",
                "head": head,
                "category": category,
                "rank": rank,
                "traj_id": int(img_key.split("_")[0]),
                "step_key": "final",
                "score": score,
            }

            manifest_path = render_exemplars(
                output_dir=tmpdir,
                trajectories=trajs,
                scores=scores,
                vae="fake_vae",
                top_k=3,
                head_names=["head_a", "head_b"],
                device="cpu",
                dtype="float32",
                deduplicate_across_heads=False,
            )

            with open(str(manifest_path)) as f:
                manifest = json.load(f)

    # Without dedup, both heads should pick the same traj_ids
    head_a_top = {e["traj_id"] for e in manifest["entries"]
                  if e["head"] == "head_a" and e["category"] == "top"}
    head_b_top = {e["traj_id"] for e in manifest["entries"]
                  if e["head"] == "head_b" and e["category"] == "top"}

    # Correlated scores -> same top-3 images
    assert head_a_top == head_b_top, (
        f"Expected same top images without dedup, got {head_a_top} vs {head_b_top}"
    )
    print("PASS: correlated heads without dedup -> same images (expected)")


def test_correlated_heads_with_dedup():
    """With deduplication, correlated heads pick DIFFERENT images."""
    from src_ii.exemplar_renderer import render_exemplars

    n = 10
    trajs = _make_fake_trajectories(n)
    scores = _make_correlated_scores(n)

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("src_ii.exemplar_renderer._render_one") as mock_render:
            mock_render.side_effect = lambda out_dir, traj_lookup, img_key, vae, device, dtype, head, rank, score, category: {
                "filename": f"{head}_{category}{rank:02d}.png",
                "head": head,
                "category": category,
                "rank": rank,
                "traj_id": int(img_key.split("_")[0]),
                "step_key": "final",
                "score": score,
            }

            manifest_path = render_exemplars(
                output_dir=tmpdir,
                trajectories=trajs,
                scores=scores,
                vae="fake_vae",
                top_k=3,
                head_names=["head_a", "head_b"],
                device="cpu",
                dtype="float32",
                deduplicate_across_heads=True,
            )

            with open(str(manifest_path)) as f:
                manifest = json.load(f)

    # With dedup, heads must pick different traj_ids
    head_a_top = {e["traj_id"] for e in manifest["entries"]
                  if e["head"] == "head_a" and e["category"] == "top"}
    head_b_top = {e["traj_id"] for e in manifest["entries"]
                  if e["head"] == "head_b" and e["category"] == "top"}

    assert len(head_a_top & head_b_top) == 0, (
        f"Expected no overlap in top images with dedup, "
        f"got overlap: {head_a_top & head_b_top}"
    )

    # head_a gets top 3: traj 9, 8, 7 (highest scores)
    assert head_a_top == {9, 8, 7}, f"Expected head_a top={{9,8,7}}, got {head_a_top}"
    # head_b gets next 3: traj 6, 5, 4 (after excluding 9, 8, 7)
    assert head_b_top == {6, 5, 4}, f"Expected head_b top={{6,5,4}}, got {head_b_top}"

    # Same for bottom
    head_a_bottom = {e["traj_id"] for e in manifest["entries"]
                     if e["head"] == "head_a" and e["category"] == "bottom"}
    head_b_bottom = {e["traj_id"] for e in manifest["entries"]
                     if e["head"] == "head_b" and e["category"] == "bottom"}

    assert len(head_a_bottom & head_b_bottom) == 0, (
        f"Expected no overlap in bottom images with dedup, "
        f"got overlap: {head_a_bottom & head_b_bottom}"
    )

    print("PASS: correlated heads with dedup -> different images")


def test_anticorrelated_heads_no_dedup_needed():
    """With anticorrelated heads, dedup changes nothing (already different)."""
    from src_ii.exemplar_renderer import render_exemplars

    n = 10
    trajs = _make_fake_trajectories(n)
    scores = _make_anticorrelated_scores(n)

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("src_ii.exemplar_renderer._render_one") as mock_render:
            mock_render.side_effect = lambda out_dir, traj_lookup, img_key, vae, device, dtype, head, rank, score, category: {
                "filename": f"{head}_{category}{rank:02d}.png",
                "head": head,
                "category": category,
                "rank": rank,
                "traj_id": int(img_key.split("_")[0]),
                "step_key": "final",
                "score": score,
            }

            manifest_path = render_exemplars(
                output_dir=tmpdir,
                trajectories=trajs,
                scores=scores,
                vae="fake_vae",
                top_k=3,
                head_names=["head_a", "head_b"],
                device="cpu",
                dtype="float32",
                deduplicate_across_heads=True,
            )

            with open(str(manifest_path)) as f:
                manifest = json.load(f)

    # With anticorrelated scores, heads naturally pick different images
    head_a_top = {e["traj_id"] for e in manifest["entries"]
                  if e["head"] == "head_a" and e["category"] == "top"}
    head_b_top = {e["traj_id"] for e in manifest["entries"]
                  if e["head"] == "head_b" and e["category"] == "top"}

    # head_a top: 9, 8, 7 (highest head_a scores)
    assert head_a_top == {9, 8, 7}, f"Expected head_a top={{9,8,7}}, got {head_a_top}"
    # head_b top: 0, 1, 2 (highest head_b scores, which are the lowest head_a)
    assert head_b_top == {0, 1, 2}, f"Expected head_b top={{0,1,2}}, got {head_b_top}"

    # No overlap (naturally)
    assert len(head_a_top & head_b_top) == 0

    print("PASS: anticorrelated heads -> naturally different images")


def test_manifest_includes_diagnostics():
    """Manifest includes rank correlation and deduplication flag."""
    from src_ii.exemplar_renderer import render_exemplars

    n = 10
    trajs = _make_fake_trajectories(n)
    scores = _make_correlated_scores(n)

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("src_ii.exemplar_renderer._render_one") as mock_render:
            mock_render.side_effect = lambda out_dir, traj_lookup, img_key, vae, device, dtype, head, rank, score, category: {
                "filename": f"{head}_{category}{rank:02d}.png",
                "head": head,
                "category": category,
                "rank": rank,
                "traj_id": int(img_key.split("_")[0]),
                "step_key": "final",
                "score": score,
            }

            manifest_path = render_exemplars(
                output_dir=tmpdir,
                trajectories=trajs,
                scores=scores,
                vae="fake_vae",
                top_k=3,
                head_names=["head_a", "head_b"],
                device="cpu",
                dtype="float32",
                deduplicate_across_heads=True,
            )

            with open(str(manifest_path)) as f:
                manifest = json.load(f)

    assert manifest["deduplicated"] is True
    assert "rank_correlations" in manifest
    assert "head_a_vs_head_b" in manifest["rank_correlations"]

    rho = manifest["rank_correlations"]["head_a_vs_head_b"]
    assert rho == 1.0, f"Expected rho=1.0 for perfectly correlated scores, got {rho}"

    assert "n_images_scored" in manifest
    assert manifest["n_images_scored"] == n

    print(f"PASS: manifest diagnostics present, rho={rho}")


def test_all_scores_saved_to_disk():
    """all_scores.json is written for post-hoc analysis."""
    from src_ii.exemplar_renderer import render_exemplars

    n = 6
    trajs = _make_fake_trajectories(n)
    scores = _make_correlated_scores(n)

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("src_ii.exemplar_renderer._render_one") as mock_render:
            mock_render.return_value = {
                "filename": "test.png",
                "head": "h",
                "category": "top",
                "rank": 0,
                "traj_id": 0,
                "step_key": "final",
                "score": 1.0,
            }

            render_exemplars(
                output_dir=tmpdir,
                trajectories=trajs,
                scores=scores,
                vae="fake_vae",
                top_k=2,
                head_names=["head_a", "head_b"],
                device="cpu",
                dtype="float32",
            )

        all_scores_path = Path(tmpdir) / "all_scores.json"
        assert all_scores_path.exists(), "all_scores.json not written"

        with open(str(all_scores_path)) as f:
            saved_scores = json.load(f)

        assert len(saved_scores) == n
        assert "0_final" in saved_scores
        assert "head_a" in saved_scores["0_final"]
        assert "head_b" in saved_scores["0_final"]

    print("PASS: all_scores.json saved to disk")


def test_rank_correlation_computation():
    """Verify Spearman rank correlation calculation."""
    from src_ii.exemplar_renderer import _compute_rank_correlation

    # Perfect correlation
    scores_corr = {
        "a": {"h1": 1.0, "h2": 10.0},
        "b": {"h1": 2.0, "h2": 20.0},
        "c": {"h1": 3.0, "h2": 30.0},
        "d": {"h1": 4.0, "h2": 40.0},
    }
    result = _compute_rank_correlation(scores_corr, ["h1", "h2"])
    assert abs(result["h1_vs_h2"] - 1.0) < 0.01, f"Expected rho~1.0, got {result}"

    # Perfect anti-correlation
    scores_anti = {
        "a": {"h1": 1.0, "h2": 40.0},
        "b": {"h1": 2.0, "h2": 30.0},
        "c": {"h1": 3.0, "h2": 20.0},
        "d": {"h1": 4.0, "h2": 10.0},
    }
    result = _compute_rank_correlation(scores_anti, ["h1", "h2"])
    assert abs(result["h1_vs_h2"] - (-1.0)) < 0.01, f"Expected rho~-1.0, got {result}"

    # Uncorrelated (approximately)
    scores_uncorr = {
        "a": {"h1": 1.0, "h2": 30.0},
        "b": {"h1": 2.0, "h2": 10.0},
        "c": {"h1": 3.0, "h2": 40.0},
        "d": {"h1": 4.0, "h2": 20.0},
    }
    result = _compute_rank_correlation(scores_uncorr, ["h1", "h2"])
    assert abs(result["h1_vs_h2"]) < 0.5, f"Expected rho~0, got {result}"

    print("PASS: rank correlation computation correct")


if __name__ == "__main__":
    test_rank_correlation_computation()
    test_correlated_heads_without_dedup()
    test_correlated_heads_with_dedup()
    test_anticorrelated_heads_no_dedup_needed()
    test_manifest_includes_diagnostics()
    test_all_scores_saved_to_disk()
    print("\nAll 6 tests passed.")

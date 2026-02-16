"""Cross-session residual analysis for trajectory latents.

Pure CPU analysis -- reads .pt files, computes statistics, writes
stats.json + matplotlib figures. No GPU, no model loading, no server.

Handles two naming conventions:
  - btrm_dataset style: step_00.pt, step_04.pt, ..., final.pt
  - stream style: euler_step_00.pt, ..., euler_step_29.pt, final_latent.pt

Compares the intersection of steps present in both directories.

Usage:
    python analyze_residuals.py DIR_A DIR_B [--out OUTPUT_DIR]

Example:
    python analyze_residuals.py \\
        btrm_dataset/latents/traj_000004 \\
        validation_renders/cross_session_latents \\
        --out validation_renders/residual_analysis
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch


def discover_steps(directory: Path) -> dict[int, Path]:
    """Find all step .pt files in a directory, return {step_idx: path}."""
    steps = {}
    for f in directory.iterdir():
        if not f.suffix == ".pt":
            continue
        # step_XX.pt or euler_step_XX.pt
        m = re.match(r"(?:euler_)?step_(\d+)\.pt$", f.name)
        if m:
            steps[int(m.group(1))] = f
        # final.pt or final_latent.pt -> map to step 30 (sentinel for "final")
        elif f.name in ("final.pt", "final_latent.pt"):
            steps[-1] = f  # -1 = final
    return steps


def load_latent(path: Path) -> np.ndarray:
    """Load a .pt latent tensor as float32 numpy array."""
    t = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(t, dict):
        # Some formats wrap in a dict
        for k in ("latent", "x", "sample"):
            if k in t:
                t = t[k]
                break
        else:
            raise ValueError(f"Cannot find latent tensor in dict keys: {list(t.keys())}")
    return t.float().numpy()


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def l2_norm(a: np.ndarray) -> float:
    return float(np.sqrt(np.sum(a ** 2)))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten()
    b_flat = b.flatten()
    dot = np.dot(a_flat, b_flat)
    na = np.linalg.norm(a_flat)
    nb = np.linalg.norm(b_flat)
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


def per_step_stats(a: np.ndarray, b: np.ndarray) -> dict:
    """Basic per-step comparison statistics."""
    residual = a - b
    l2_res = l2_norm(residual)
    l2_ref = l2_norm(a)
    return {
        "mse": mse(a, b),
        "l2_residual": l2_res,
        "l2_reference": l2_ref,
        "relative_error": float(l2_res / l2_ref) if l2_ref > 0 else float("inf"),
        "cosine_similarity": cosine_similarity(a, b),
        "max_abs_diff": float(np.max(np.abs(residual))),
        "mean_abs_diff": float(np.mean(np.abs(residual))),
    }


def channel_covariance_analysis(residual: np.ndarray) -> dict:
    """Analyze spatial covariance structure across channels.

    residual: shape (1, C, H, W) or (C, H, W)
    Returns correlation matrix eigenspectrum and top correlations.
    """
    if residual.ndim == 4:
        residual = residual[0]  # Remove batch dim
    C, H, W = residual.shape

    # Flatten each channel to a vector
    flat = residual.reshape(C, -1)  # (C, H*W)

    # Channel correlation matrix
    # Normalize each channel to zero mean, unit variance
    means = flat.mean(axis=1, keepdims=True)
    stds = flat.std(axis=1, keepdims=True)
    stds = np.where(stds < 1e-10, 1.0, stds)
    normed = (flat - means) / stds

    corr_matrix = (normed @ normed.T) / flat.shape[1]  # (C, C)

    # Eigenspectrum
    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending

    # Effective rank (participation ratio)
    eigenvalues_pos = eigenvalues[eigenvalues > 0]
    if len(eigenvalues_pos) > 0:
        p = eigenvalues_pos / eigenvalues_pos.sum()
        entropy = -np.sum(p * np.log(p + 1e-15))
        effective_rank = float(np.exp(entropy))
    else:
        effective_rank = 0.0

    return {
        "eigenspectrum": eigenvalues.tolist(),
        "effective_rank": effective_rank,
        "top_3_eigenvalues": eigenvalues[:3].tolist(),
        "trace": float(np.trace(corr_matrix)),
        "per_channel_std": stds.flatten().tolist(),
        "per_channel_mean": means.flatten().tolist(),
    }


def _numpy_skewness(x: np.ndarray) -> float:
    """Skewness using numpy only (Fisher definition)."""
    n = len(x)
    m = x.mean()
    s = x.std(ddof=0)
    if s < 1e-15:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))


def _numpy_kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis using numpy only (0 for Gaussian)."""
    n = len(x)
    m = x.mean()
    s = x.std(ddof=0)
    if s < 1e-15:
        return 0.0
    return float(np.mean(((x - m) / s) ** 4) - 3.0)


try:
    from scipy import stats as sp_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def gaussianity_analysis(residual: np.ndarray, max_samples: int = 5000) -> dict:
    """Test whether the residual is Gaussian-distributed per channel.

    Computes skewness, kurtosis. Shapiro-Wilk if scipy is available.
    """
    if residual.ndim == 4:
        residual = residual[0]
    C, H, W = residual.shape

    results = {"per_channel": [], "has_scipy": _HAS_SCIPY}
    all_skew = []
    all_kurt = []
    all_shapiro_p = []

    rng = np.random.default_rng(42)

    for c in range(C):
        chan = residual[c].flatten()
        n = len(chan)

        skewness = _numpy_skewness(chan)
        kurtosis = _numpy_kurtosis(chan)

        entry = {
            "channel": c,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "std": float(np.std(chan)),
            "mean": float(np.mean(chan)),
        }

        if _HAS_SCIPY:
            if n > max_samples:
                subsample = rng.choice(chan, size=max_samples, replace=False)
            else:
                subsample = chan
            sw_stat, sw_p = sp_stats.shapiro(subsample)
            entry["shapiro_w"] = float(sw_stat)
            entry["shapiro_p"] = float(sw_p)
            all_shapiro_p.append(float(sw_p))

        all_skew.append(skewness)
        all_kurt.append(kurtosis)
        results["per_channel"].append(entry)

    summary = {
        "mean_abs_skewness": float(np.mean(np.abs(all_skew))),
        "mean_excess_kurtosis": float(np.mean(all_kurt)),
        "total_channels": C,
    }
    if all_shapiro_p:
        summary["min_shapiro_p"] = float(np.min(all_shapiro_p))
        summary["median_shapiro_p"] = float(np.median(all_shapiro_p))
        summary["channels_gaussian_p05"] = int(np.sum(np.array(all_shapiro_p) > 0.05))
    results["summary"] = summary
    return results


def spectral_analysis(residual: np.ndarray) -> dict:
    """2D FFT power spectrum of residual to detect spatial autocorrelation.

    If residual is spatially white noise, power spectrum should be flat.
    Structured artifacts show up as spectral peaks.
    """
    if residual.ndim == 4:
        residual = residual[0]
    C, H, W = residual.shape

    # Average power spectrum across channels
    power_sum = np.zeros((H, W))
    for c in range(C):
        fft2 = np.fft.fft2(residual[c])
        power = np.abs(fft2) ** 2
        power_sum += power

    power_avg = power_sum / C

    # Shift zero-frequency to center
    power_centered = np.fft.fftshift(power_avg)

    # Radial average (azimuthal integration)
    cy, cx = H // 2, W // 2
    max_r = min(cy, cx)
    radial_bins = np.zeros(max_r)
    radial_counts = np.zeros(max_r)

    y_grid, x_grid = np.ogrid[:H, :W]
    r_grid = np.sqrt((y_grid - cy) ** 2 + (x_grid - cx) ** 2).astype(int)

    for r in range(max_r):
        mask = r_grid == r
        radial_bins[r] = power_centered[mask].mean() if mask.any() else 0
        radial_counts[r] = mask.sum()

    # Flatness metric: std/mean of radial profile (0 = perfectly flat = white noise)
    radial_nonzero = radial_bins[radial_bins > 0]
    if len(radial_nonzero) > 1:
        flatness = float(np.std(radial_nonzero) / np.mean(radial_nonzero))
    else:
        flatness = 0.0

    # Peak-to-DC ratio: max non-DC bin / DC bin
    if radial_bins[0] > 0 and len(radial_bins) > 1:
        peak_to_dc = float(np.max(radial_bins[1:]) / radial_bins[0])
    else:
        peak_to_dc = 0.0

    return {
        "radial_power_spectrum": radial_bins.tolist(),
        "spectral_flatness": flatness,
        "peak_to_dc_ratio": peak_to_dc,
        "dc_power": float(radial_bins[0]),
        "power_spectrum_2d": power_centered.tolist(),
    }


def plot_results(stats: dict, out_dir: Path):
    """Generate matplotlib figures from computed statistics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. Per-step MSE and relative error
    steps = sorted(stats["per_step"].keys(), key=lambda x: int(x))
    step_nums = [int(s) for s in steps]
    mses = [stats["per_step"][s]["mse"] for s in steps]
    rel_errs = [stats["per_step"][s]["relative_error"] for s in steps]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.semilogy(step_nums, mses, "o-")
    ax1.set_xlabel("Euler Step")
    ax1.set_ylabel("MSE")
    ax1.set_title("Per-Step MSE (A vs B)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(step_nums, rel_errs, "o-", color="tab:orange")
    ax2.set_xlabel("Euler Step")
    ax2.set_ylabel("Relative L2 Error")
    ax2.set_title("Per-Step Relative Error")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "per_step_mse.png", dpi=150)
    plt.close(fig)

    # 2. Channel covariance eigenspectrum (final step)
    final_key = steps[-1]
    if "covariance" in stats["per_step"][final_key]:
        eigs = stats["per_step"][final_key]["covariance"]["eigenspectrum"]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.semilogy(range(len(eigs)), eigs, "o-")
        ax.set_xlabel("Eigenvalue Index")
        ax.set_ylabel("Eigenvalue (log)")
        ax.set_title(f"Channel Correlation Eigenspectrum (step {final_key})")
        ax.grid(True, alpha=0.3)
        eff_rank = stats["per_step"][final_key]["covariance"]["effective_rank"]
        ax.axhline(y=eigs[0] / len(eigs), color="r", linestyle="--", alpha=0.5,
                   label=f"Effective rank: {eff_rank:.1f}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "eigenspectrum.png", dpi=150)
        plt.close(fig)

    # 3. Gaussianity: skewness/kurtosis per channel (final step)
    if "gaussianity" in stats["per_step"][final_key]:
        gauss = stats["per_step"][final_key]["gaussianity"]["per_channel"]
        channels = [g["channel"] for g in gauss]
        skews = [g["skewness"] for g in gauss]
        kurts = [g["kurtosis"] for g in gauss]
        has_shapiro = "shapiro_p" in gauss[0]
        n_cols = 3 if has_shapiro else 2

        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
        axes[0].bar(channels, skews, alpha=0.7)
        axes[0].axhline(y=0, color="r", linestyle="--", alpha=0.5)
        axes[0].set_xlabel("Channel")
        axes[0].set_ylabel("Skewness")
        axes[0].set_title("Per-Channel Skewness")

        axes[1].bar(channels, kurts, alpha=0.7, color="tab:orange")
        axes[1].axhline(y=0, color="r", linestyle="--", alpha=0.5)
        axes[1].set_xlabel("Channel")
        axes[1].set_ylabel("Excess Kurtosis")
        axes[1].set_title("Per-Channel Kurtosis")

        if has_shapiro:
            shapiro_ps = [g["shapiro_p"] for g in gauss]
            axes[2].bar(channels, shapiro_ps, alpha=0.7, color="tab:green")
            axes[2].axhline(y=0.05, color="r", linestyle="--", alpha=0.5, label="p=0.05")
            axes[2].set_xlabel("Channel")
            axes[2].set_ylabel("Shapiro-Wilk p-value")
            axes[2].set_title("Gaussianity Test")
            axes[2].legend()

        fig.tight_layout()
        fig.savefig(out_dir / "gaussianity.png", dpi=150)
        plt.close(fig)

    # 4. Radial power spectrum (final step)
    if "spectral" in stats["per_step"][final_key]:
        radial = stats["per_step"][final_key]["spectral"]["radial_power_spectrum"]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.semilogy(range(len(radial)), radial, "-")
        ax.set_xlabel("Spatial Frequency (radial)")
        ax.set_ylabel("Power (log)")
        ax.set_title(f"Radial Power Spectrum of Residual (step {final_key})")
        ax.grid(True, alpha=0.3)
        flatness = stats["per_step"][final_key]["spectral"]["spectral_flatness"]
        ax.text(0.95, 0.95, f"Flatness: {flatness:.3f}\n(0 = white noise)",
                transform=ax.transAxes, ha="right", va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        fig.tight_layout()
        fig.savefig(out_dir / "radial_power_spectrum.png", dpi=150)
        plt.close(fig)

    # 5. Spatial residual heatmap (final step, channel-averaged)
    # This requires the actual residual data, which we don't have in stats.
    # Instead we'll plot per-channel std from covariance analysis.
    if "covariance" in stats["per_step"][final_key]:
        stds = stats["per_step"][final_key]["covariance"]["per_channel_std"]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(stds)), stds, alpha=0.7)
        ax.set_xlabel("Channel")
        ax.set_ylabel("Residual Std Dev")
        ax.set_title(f"Per-Channel Residual Magnitude (step {final_key})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "per_channel_residual_std.png", dpi=150)
        plt.close(fig)

    print(f"Figures saved to {out_dir}/")


def pretty_print_summary(stats: dict, out_path: Path):
    """Write a human/LLM-readable summary of the residual analysis.

    This is the primary readout. stats.json is for programmatic access;
    this file is for understanding what the numbers mean.
    """
    lines = []
    w = lines.append

    w("=" * 72)
    w("RESIDUAL ANALYSIS SUMMARY")
    w("=" * 72)
    w(f"Reference (A): {stats['dir_a']}")
    w(f"Comparison (B): {stats['dir_b']}")
    w(f"Common steps: {stats['common_steps']}")
    w("")

    # --- Per-step trajectory table ---
    w("--- PER-STEP METRICS ---")
    w(f"{'Step':>6}  {'MSE':>12}  {'Rel Err':>10}  {'Max|Diff|':>10}  {'Mean|Diff|':>10}")
    w("-" * 60)

    step_labels = list(stats["per_step"].keys())
    for label in step_labels:
        s = stats["per_step"][label]
        if "error" in s:
            w(f"{label:>6}  {s['error']}")
            continue
        w(f"{label:>6}  {s['mse']:>12.4e}  {s['relative_error']:>10.4f}  "
          f"{s['max_abs_diff']:>10.4f}  {s['mean_abs_diff']:>10.4f}")
    w("")

    # --- MSE trajectory characterization ---
    mses = [stats["per_step"][l]["mse"] for l in step_labels if "mse" in stats["per_step"][l]]
    if len(mses) >= 2:
        mse_trend = mses[-1] / mses[0] if mses[0] > 0 else float("inf")
        if mse_trend > 2.0:
            w(f"MSE GROWS {mse_trend:.1f}x from first to last step: divergence accumulates.")
        elif mse_trend < 0.5:
            w(f"MSE SHRINKS {mse_trend:.2f}x from first to last step: convergent trajectories.")
        else:
            w(f"MSE ratio first-to-last: {mse_trend:.2f}x (stable magnitude).")
        w("")

    # --- Covariance analysis for endpoint steps ---
    for label in step_labels:
        s = stats["per_step"][label]
        if "covariance" not in s:
            continue

        cov = s["covariance"]
        w(f"--- COVARIANCE STRUCTURE (step {label}) ---")

        eff_rank = cov["effective_rank"]
        n_channels = len(cov["eigenspectrum"])
        top3 = cov["top_3_eigenvalues"]
        top3_frac = sum(top3) / cov["trace"] if cov["trace"] > 0 else 0

        w(f"  Effective rank: {eff_rank:.1f} / {n_channels} channels")
        w(f"  Top 3 eigenvalues: {top3[0]:.2f}, {top3[1]:.2f}, {top3[2]:.2f} "
          f"({top3_frac:.0%} of variance)")

        if eff_rank < n_channels * 0.4:
            w(f"  INTERPRETATION: Low-rank residual (eff_rank={eff_rank:.1f} << {n_channels}).")
            w(f"    The difference lives in {int(eff_rank)}-dimensional subspace.")
            w(f"    This is STRUCTURED divergence, not diffuse noise.")
        elif eff_rank > n_channels * 0.8:
            w(f"  INTERPRETATION: High-rank residual (eff_rank={eff_rank:.1f} ~ {n_channels}).")
            w(f"    Difference is spread across channels.")
            w(f"    This looks like DIFFUSE noise, not structured divergence.")
        else:
            w(f"  INTERPRETATION: Moderate rank ({eff_rank:.1f}/{n_channels}).")
            w(f"    Mix of structured and diffuse components.")

        # Channel heterogeneity
        stds = cov["per_channel_std"]
        std_ratio = max(stds) / min(stds) if min(stds) > 0 else float("inf")
        max_ch = stds.index(max(stds))
        min_ch = stds.index(min(stds))
        w(f"  Channel std range: {min(stds):.3f} (ch{min_ch}) to {max(stds):.3f} (ch{max_ch}), "
          f"ratio={std_ratio:.1f}x")
        if std_ratio > 3.0:
            w(f"  WARNING: Channel {max_ch} carries {std_ratio:.0f}x more residual than ch{min_ch}.")
            w(f"    One latent channel dominates the divergence.")
        w("")

    # --- Gaussianity analysis for endpoint steps ---
    for label in step_labels:
        s = stats["per_step"][label]
        if "gaussianity" not in s:
            continue

        gauss = s["gaussianity"]
        summary = gauss["summary"]
        w(f"--- GAUSSIANITY (step {label}) ---")
        w(f"  Mean |skewness|: {summary['mean_abs_skewness']:.3f} "
          f"(Gaussian: ~0, concern: >0.5)")
        w(f"  Mean excess kurtosis: {summary['mean_excess_kurtosis']:.3f} "
          f"(Gaussian: 0, heavy tails: >0)")

        # Flag outlier channels
        outliers = []
        for ch in gauss["per_channel"]:
            if abs(ch["skewness"]) > 0.5 or abs(ch["kurtosis"]) > 0.5:
                outliers.append(ch)
        if outliers:
            w(f"  OUTLIER CHANNELS (|skew|>0.5 or |kurt|>0.5):")
            for o in outliers:
                w(f"    ch{o['channel']}: skew={o['skewness']:.3f}, "
                  f"kurt={o['kurtosis']:.3f}, std={o['std']:.3f}")
        else:
            w(f"  All channels within Gaussian envelope (|skew|<0.5, |kurt|<0.5).")

        if "channels_gaussian_p05" in summary:
            w(f"  Shapiro-Wilk: {summary['channels_gaussian_p05']}/{summary['total_channels']} "
              f"channels pass p>0.05")

        # Overall assessment
        if summary["mean_abs_skewness"] < 0.2 and abs(summary["mean_excess_kurtosis"]) < 0.3:
            w(f"  ASSESSMENT: Residual is approximately Gaussian per channel.")
            w(f"    Consistent with accumulated floating-point rounding noise.")
        elif summary["mean_abs_skewness"] > 0.5 or abs(summary["mean_excess_kurtosis"]) > 1.0:
            w(f"  ASSESSMENT: Residual is NON-Gaussian.")
            w(f"    This suggests structured model divergence, not FP noise.")
        else:
            w(f"  ASSESSMENT: Mildly non-Gaussian. Could be either FP noise with")
            w(f"    channel-dependent structure, or weak structural divergence.")
        w("")

    # --- Spectral analysis for endpoint steps ---
    for label in step_labels:
        s = stats["per_step"][label]
        if "spectral" not in s:
            continue

        spec = s["spectral"]
        w(f"--- SPATIAL SPECTRUM (step {label}) ---")
        w(f"  Spectral flatness: {spec['spectral_flatness']:.2f} "
          f"(0 = white noise, >2 = spatially structured)")
        w(f"  Peak-to-DC ratio: {spec['peak_to_dc_ratio']:.3f}")

        radial = spec["radial_power_spectrum"]
        if len(radial) > 5:
            # Check for 1/f-like falloff
            low_band = np.mean(radial[1:5]) if len(radial) > 4 else radial[1]
            high_band = np.mean(radial[-5:])
            lh_ratio = low_band / high_band if high_band > 0 else float("inf")
            w(f"  Low-freq / high-freq power ratio: {lh_ratio:.1f}x")

            if lh_ratio > 10:
                w(f"  INTERPRETATION: Strong low-frequency dominance.")
                w(f"    Residual has large-scale spatial structure (blobs, gradients).")
                w(f"    NOT consistent with pixel-independent noise.")
            elif lh_ratio < 2:
                w(f"  INTERPRETATION: Relatively flat spectrum.")
                w(f"    Residual is spatially uncorrelated (white-noise-like).")
            else:
                w(f"  INTERPRETATION: Moderate spectral tilt ({lh_ratio:.0f}x).")
                w(f"    Some low-frequency structure, but not dominant.")
        w("")

    # --- Overall verdict ---
    w("=" * 72)
    w("OVERALL CHARACTERIZATION")
    w("=" * 72)

    # Synthesize across all endpoint analyses
    has_low_rank = False
    has_non_gaussian = False
    has_spatial_structure = False

    for label in step_labels:
        s = stats["per_step"][label]
        if "covariance" in s:
            n_ch = len(s["covariance"]["eigenspectrum"])
            if s["covariance"]["effective_rank"] < n_ch * 0.5:
                has_low_rank = True
        if "gaussianity" in s:
            sm = s["gaussianity"]["summary"]
            if sm["mean_abs_skewness"] > 0.3 or abs(sm["mean_excess_kurtosis"]) > 0.5:
                has_non_gaussian = True
        if "spectral" in s:
            if s["spectral"]["spectral_flatness"] > 3.0:
                has_spatial_structure = True

    if has_low_rank and has_non_gaussian and has_spatial_structure:
        w("These trajectories show STRUCTURED DIVERGENCE:")
        w("  - Low-rank channel covariance (semantic subspace)")
        w("  - Non-Gaussian residual distribution")
        w("  - Spatially correlated differences")
        w("This is consistent with DIFFERENT SEMANTIC CONTENT,")
        w("not numerical noise from the same generative process.")
    elif not has_low_rank and not has_non_gaussian and not has_spatial_structure:
        w("These trajectories show NOISE-LIKE DIVERGENCE:")
        w("  - High-rank (diffuse) channel covariance")
        w("  - Gaussian residual distribution")
        w("  - Spatially uncorrelated (white noise)")
        w("This is consistent with floating-point accumulation noise")
        w("from two runs of the SAME generative process.")
    else:
        w("MIXED SIGNAL:")
        if has_low_rank:
            w("  - Low-rank covariance suggests structured divergence")
        else:
            w("  - High-rank covariance suggests diffuse noise")
        if has_non_gaussian:
            w("  - Non-Gaussian residual suggests structured divergence")
        else:
            w("  - Gaussian residual suggests FP noise")
        if has_spatial_structure:
            w("  - Spatial structure suggests correlated differences")
        else:
            w("  - Flat spectrum suggests pixel-independent noise")
        w("Further investigation needed to disambiguate.")
    w("")

    # --- Artifact inventory ---
    w("--- ARTIFACTS ON DISK ---")
    for label in step_labels:
        s = stats["per_step"][label]
        if "residual_saved" in s:
            w(f"  {s['residual_saved']}")
    w(f"  {out_path}")
    w(f"  {out_path.parent / 'stats.json'}")
    w("")

    text = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(text)
    # Also print to stdout
    print(text)
    return text


def analyze(dir_a: Path, dir_b: Path, out_dir: Path):
    """Run full residual analysis on two trajectory directories."""
    steps_a = discover_steps(dir_a)
    steps_b = discover_steps(dir_b)

    common_steps = sorted(set(steps_a.keys()) & set(steps_b.keys()))
    if not common_steps:
        print(f"No common steps found between {dir_a} and {dir_b}")
        print(f"  Dir A steps: {sorted(steps_a.keys())}")
        print(f"  Dir B steps: {sorted(steps_b.keys())}")
        sys.exit(1)

    print(f"Dir A: {dir_a} ({len(steps_a)} steps)")
    print(f"Dir B: {dir_b} ({len(steps_b)} steps)")
    print(f"Common steps: {common_steps}")
    print()

    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "dir_a": str(dir_a),
        "dir_b": str(dir_b),
        "common_steps": common_steps,
        "per_step": {},
    }

    for step in common_steps:
        label = str(step) if step >= 0 else "final"
        print(f"Step {label}...", end=" ", flush=True)

        a = load_latent(steps_a[step])
        b = load_latent(steps_b[step])

        if a.shape != b.shape:
            print(f"SHAPE MISMATCH: {a.shape} vs {b.shape}")
            stats["per_step"][label] = {"error": f"shape mismatch: {a.shape} vs {b.shape}"}
            continue

        step_stats = per_step_stats(a, b)

        residual = a - b

        # Full analysis on first, last, and final steps
        is_endpoint = (step == common_steps[0] or step == common_steps[-1] or step == -1)
        if is_endpoint:
            step_stats["covariance"] = channel_covariance_analysis(residual)
            step_stats["gaussianity"] = gaussianity_analysis(residual)
            step_stats["spectral"] = spectral_analysis(residual)

            # Save the residual tensor for downstream inspection
            residual_path = out_dir / f"residual_{label}.npy"
            np.save(residual_path, residual)
            step_stats["residual_saved"] = str(residual_path)

        print(f"MSE={step_stats['mse']:.2e}  rel_err={step_stats['relative_error']:.4f}  "
              f"cos={step_stats['cosine_similarity']:.6f}")
        stats["per_step"][label] = step_stats

    # Remove 2D power spectrum from JSON (too large, keep in .npy)
    for label, s in stats["per_step"].items():
        if "spectral" in s and "power_spectrum_2d" in s["spectral"]:
            ps2d = np.array(s["spectral"].pop("power_spectrum_2d"))
            np.save(out_dir / f"power_spectrum_2d_{label}.npy", ps2d)

    # Write stats
    stats_path = out_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats written to {stats_path}")

    # Pretty-print summary (always works, no optional deps)
    summary_path = out_dir / "summary.txt"
    pretty_print_summary(stats, summary_path)

    # Generate plots
    try:
        plot_results(stats, out_dir)
    except ImportError as e:
        print(f"Skipping plots (missing dependency): {e}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Analyze residuals between two trajectory latent directories."
    )
    parser.add_argument("dir_a", type=Path, help="First trajectory directory (reference)")
    parser.add_argument("dir_b", type=Path, help="Second trajectory directory (comparison)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output directory for stats and figures (default: dir_b/residual_analysis)")
    args = parser.parse_args()

    if not args.dir_a.is_dir():
        print(f"Error: {args.dir_a} is not a directory")
        sys.exit(1)
    if not args.dir_b.is_dir():
        print(f"Error: {args.dir_b} is not a directory")
        sys.exit(1)

    out_dir = args.out or args.dir_b / "residual_analysis"
    analyze(args.dir_a, args.dir_b, out_dir)


if __name__ == "__main__":
    main()

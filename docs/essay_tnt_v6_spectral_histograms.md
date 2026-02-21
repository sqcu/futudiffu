# TNT v6: Multi-Scale Chebyshev Spectral Descriptor Histograms

## Motivation

All previous TNT versions (v1 through v5) have been validated. The recurring finding: the problem is **discrimination**, not rotation invariance per se. v5 (Zernike moments of Fiedler) achieved 9/12 fractional rotation passes but only 2/5 constraints because the two anchor images are nearly indistinguishable in moment space (distance 0.013 between different images vs 0.023 for self-rotation, ratio 0.56). Every method so far collapsed spatial information too early -- into a single Fiedler vector, a set of moment magnitudes, or segment-level signatures -- losing the per-pixel richness needed to tell pizza-ratto from offhand_pleometric.

The hypothesis for v6: compute **rich per-pixel descriptors** via multi-scale graph spectral filtering, then aggregate via **permutation-invariant statistics** (histograms). Rotation changes WHERE pixels are but not the DISTRIBUTION of their spectral descriptors.

## Method

### Why Chebyshev Polynomials

Raw iterated powers L^k of the graph Laplacian suffer from numerical instability: eigenvalues in [0, lambda_max] get raised to the k-th power, causing the largest eigenvalues to dominate exponentially. After just a few iterations, all information about the smooth/mid-frequency structure is lost.

Chebyshev polynomials T_k(x) are the optimal polynomial basis for [-1, 1] in the minimax sense. By rescaling the Laplacian to L_tilde = (2/lambda_max)*L - I, mapping eigenvalues to [-1, 1], the Chebyshev recurrence:

- T_0(L~)x = x
- T_1(L~)x = L~ @ x
- T_k(L~)x = 2 * L~ @ T_{k-1} - T_{k-2}

produces a well-conditioned filter bank where each order k acts as a spectral bandpass filter. T_0 is the original signal, T_1 isolates mid-frequency content, and higher k isolate progressively higher frequencies. The recurrence stores only the last two iterates, making it memory-efficient.

### Why Per-Pixel Descriptors + Histograms

The key insight: if we treat the RGB image as 3 probe vectors applied through the Chebyshev filter bank, each pixel gets a K*3-dimensional descriptor encoding its spectral characteristics at multiple scales. Two images with the same scene structure but different pixel orderings (e.g., rotation) will have the same multiset of descriptors.

We aggregate by computing the **energy** (squared L2 norm over RGB) at each scale per pixel, then computing a **histogram** of energy values across all pixels. The (K, num_bins) histogram bank is the final descriptor. This is:
- Permutation-invariant: rotating pixels does not change the histogram
- Discriminative: structurally different images produce different energy distributions
- Computationally cheap: K wide matvecs + histogram binning

### Distance Metrics

We tested two distance metrics on the histogram banks:
- **Chi-squared**: chi2 = sum_k sum_b (h1[k,b] - h2[k,b])^2 / (h1[k,b] + h2[k,b] + eps). Emphasizes bins with significant mass.
- **L2**: simple Euclidean distance between flattened histogram banks.

And two binning modes:
- **Log-spaced**: log-transform energy before uniform binning (better for heavy-tailed distributions)
- **Linear**: direct uniform binning on raw energy values

## Results

### Constraint Results (Part A)

5 TNT constraints checked on all 11 images in i2i_off_policies/:

| Method | 1: THIS>THAT | 2: THIS>ALL | 3: THAT<ALL | 4: SKETCH>COLOR | 5: THAT<NIGHT | Total |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| v1_gpu | PASS | PASS | PASS | FAIL | PASS | 4/5 |
| v2_lanczos | PASS | PASS | PASS | PASS | PASS | 5/5 |
| v5_zernike_fiedler | PASS | FAIL | FAIL | FAIL | PASS | 2/5 |
| **v6_spectral_hist** | **PASS** | **PASS** | **PASS** | **PASS** | **PASS** | **5/5** |

v6 achieves perfect constraint satisfaction, matching v2 and dramatically improving on v5. The core discrimination is strong: pizza-ratto scores +0.929, offhand_pleometric scores -0.929, and all other images fall between these extremes.

### Invariance Results (Part B)

Full invariance suite: 10 basic transforms + 6 fractional rotations (border padding), 2 anchors.

**Overall pass rates:**

| Method | pizza-ratto | offhand_pleometric | Overall |
|--------|:---:|:---:|:---:|
| v1_gpu | 94% | 94% | 30/32 |
| v2_lanczos | 100% | 62% | 26/32 |
| v5_zernike_fiedler | 62% | 100% | 26/32 |
| v6_spectral_hist | 38% | 94% | 21/32 |

**Fractional rotation results (sign preservation):**

| Method | Passed | Failed |
|--------|:---:|:---:|
| v1_gpu | 12/12 | 0 |
| v2_lanczos | 7/12 | 5 (all offhand) |
| v5_zernike_fiedler | 9/12 | 3 (pizza-ratto: rot72, rot15, rot7) |
| v6_spectral_hist | 6/12 | 6 (all pizza-ratto rotations) |

v6 fails all 6 fractional rotations for pizza-ratto but passes all 6 for offhand_pleometric. The failure mode is specific: border-padding rotation on pizza-ratto perturbs the energy histogram enough to flip the sign. For offhand_pleometric, the identity distance to THAT is large enough (-0.929) that rotation perturbation (-0.085 to -0.145) stays negative.

### Discrimination Diagnostic (Part C)

This is the critical new test. For each combination of bin_mode and distance_mode, we compute:
- Cross-image distance: dist(pizza-ratto, offhand_pleometric)
- Self-rotation distance: dist(pizza-ratto, rotated_72_pizza-ratto)
- Discrimination ratio: cross / self_rot (should be >> 1)

| Config | Cross dist | Self-rot (pizza) | Ratio (pizza) | Self-rot (offhand) | Ratio (offhand) |
|--------|:---:|:---:|:---:|:---:|:---:|
| log/chi2 | 13.12 | 14.14 | 0.93 | 5.63 | 2.33 |
| log/l2 | 2.93 | 3.45 | 0.85 | 1.68 | 1.74 |
| linear/chi2 | 15.99 | 16.02 | 1.00 | 11.64 | 1.37 |
| linear/l2 | 3.06 | 3.41 | 0.90 | 2.34 | 1.31 |

**Comparison to v5:** v5 had ratio 0.56 (cross=0.013, self-rot=0.023). v6 ratios for pizza-ratto range from 0.85 to 1.00 -- still below 1.0, meaning the self-rotation perturbation exceeds the cross-image gap for pizza-ratto. However, v6 ratios for offhand_pleometric are 1.31 to 2.33, much better than v5.

The asymmetry (pizza-ratto ratio < 1, offhand_pleometric ratio > 1) explains the Part B results: offhand_pleometric rotations preserve sign because the cross-image gap is large relative to rotation perturbation, while pizza-ratto rotations flip sign because they don't.

### False-Color Analysis

The filter bank visualizations (tnt_v6_validation/visualizations/) show:

1. **Scale k=0** (T_0): the original RGB image -- identical for original and rotated
2. **Scale k=2**: mid-frequency spatial structure -- contours, edges, texture patterns
3. **Scale k=5**: higher-frequency detail -- fine texture, noise-level features
4. **Scale k=9**: highest frequency band -- captures pixel-level irregularity

The energy histograms at each scale show that the original and rotated pizza-ratto have visibly different distributions, especially at higher scales (k=5, k=9), where border-padding resampling artifacts contribute energy at different scales than the original image structure. The offhand_pleometric energy distributions are clearly distinguishable from pizza-ratto at all scales.

The histogram bank heatmaps confirm: pizza-ratto vs rotated pizza-ratto show moderate differences across all scales, while pizza-ratto vs offhand_pleometric show stark differences in the distribution shape.

### Timing Comparison

| Method | Avg ms/image |
|--------|:---:|
| v1_gpu | 7.7 |
| v6_spectral_hist | 35.9 |
| v5_zernike_fiedler | 79.4 |
| v2_lanczos | 648.2 |

v6 is 2.2x faster than v5 and 18x faster than v2. The K=10 Chebyshev matvecs are cheaper than the 30-iteration compiled Lanczos + Zernike moment computation.

## Assessment

### What v6 Gets Right

1. **Discrimination (5/5 constraints)**: The histogram-based descriptor separates all 11 images correctly, including the hardest case (THIS vs THAT separation: 0.929 vs -0.929, a gap of 1.858 vs v5's gap of 0.025).

2. **Basic invariance**: Horizontal flip, rotate 90, and hue shift achieve perfect score preservation. The graph Laplacian is symmetric under pixel permutation, and hue shift doesn't change the graph structure (same edge weights when threshold is color-distance based).

3. **Speed**: 36ms average, practical for real-time scoring.

### What v6 Gets Wrong

1. **Fractional rotation tolerance for pizza-ratto**: 0/6 fractional rotations preserve sign for pizza-ratto. The root cause is that border-padding rotation introduces edge-replicated uniform regions at the image corners, which create a new population of low-energy pixels at all scales. This shifts the histogram in a way that exceeds the inter-image distance.

2. **Noise sensitivity**: Gaussian noise (sigma=0.05) flips sign for pizza-ratto. The noise changes the energy distribution at high scales.

3. **Geometric transforms**: Shear, scale+pad, and center_crop all flip sign for pizza-ratto. These transforms change the pixel population (zero/reflect padding, interpolation artifacts) enough to perturb the histogram.

### The Fundamental Tension

v6 demonstrates the core tension between **discrimination** and **rotation tolerance** for this problem:

- **High discrimination** requires the descriptor to be sensitive to the differences between pizza-ratto and offhand_pleometric. v6's histograms capture these differences at 0.929 separation.
- **Rotation tolerance** requires the descriptor to be insensitive to the changes that rotation introduces (border padding, bilinear interpolation, coordinate rounding). v6's histograms change by distances of 3-14 under rotation.

When the rotation-induced distance (3-14) exceeds the cross-image distance (3-16), discrimination fails. The problem is that for pizza-ratto specifically, rotation perturbation and cross-image difference are in the same magnitude range.

### Does This Finally Get Both?

No. v6 achieves excellent discrimination (5/5 constraints, 1.858 gap) but poor rotation tolerance for pizza-ratto (0/6 rotations). This is a significant improvement over v5's discrimination (2/5 constraints, 0.025 gap) but a regression in rotation tolerance (v5 got 9/12).

The approaches explored so far can be mapped on a discrimination-invariance Pareto frontier:
- **v1** (pixel cosine): low discrimination (4/5), high invariance (30/32)
- **v2** (Lanczos segments): high discrimination (5/5), moderate invariance (26/32, fails offhand rotations)
- **v5** (Zernike of Fiedler): low discrimination (2/5), good rotation invariance (9/12 rotations)
- **v6** (Chebyshev histograms): high discrimination (5/5), poor rotation invariance (6/12 rotations)

No method achieves both. The next step would be to either:
1. Make the histogram comparison robust to rotation artifacts (e.g., exclude border pixels, use wasserstein distance instead of chi2/L2)
2. Combine v5's rotation invariance with v6's discrimination in an ensemble
3. Pre-process rotated images to remove border artifacts before scoring

## Files

- Core functions: `src_ii/itten_cuter_grops.py` (chebyshev_filter_bank, spectral_energy_histograms, spectral_histogram_similarity)
- Scoring function: `src_ii/reward_functions.py` (thisnotthat_score_v6)
- Validation script: `scripts_ii/validate_tnt_v6_spectral_histograms.py`
- Results: `tnt_v6_validation/` (part_a_scores.json, part_b_invariance.json, part_c_discrimination.json, summary.json)
- Visualizations: `tnt_v6_validation/visualizations/` (filter bank panels, energy histograms, histogram bank heatmaps)

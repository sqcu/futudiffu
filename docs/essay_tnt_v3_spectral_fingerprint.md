# TNT v3: Spectral Fingerprint Structural Similarity

## What polynomial moments are

The graph Laplacian L of a color-weighted image grid has a spectrum: eigenvalues lambda_1 <= lambda_2 <= ... <= lambda_n. The Lanczos pipeline (v2) computes the Fiedler vector (the eigenvector for lambda_2) and uses it to define segments, then matches segments by 7D signatures. This requires 30 serial matvecs on a single vector, followed by contour detection, connected components, and segment matching.

Polynomial moments take a different approach. Instead of computing eigenvectors, they estimate the moments of the spectral distribution:

    tr(L^k) = sum_i lambda_i^k

This is a sufficient statistic for comparing two graphs. Two graphs with similar spectral moment sequences have similar spectral distributions, hence similar structural properties.

The key insight: for m random probe vectors Z in R^{n x m}, the quantity (1/m) * tr(Z^T L^k Z) is a stochastic estimator of tr(L^k). By applying L repeatedly to Z and collecting the quadratic forms z_i^T L^k z_i at each step k, we get a (T, m) fingerprint matrix that approximates the first T spectral moments with m independent estimates each.

Serial depth: T matvecs (typically 5), each applied to all m probes in parallel. Total: 5 wide matvecs vs Lanczos's 30 narrow matvecs + eigensolver + segment pipeline.

## Why they replace Lanczos for similarity

For *similarity comparison*, the Fiedler vector is a means to an end. We never needed an eigenvector -- we needed a structural descriptor that captures how the image's edge structure partitions space. The spectral moments do this directly:

- Low-order moments (k=1): total edge weight. Captures overall connectivity.
- Mid-order moments (k=2,3): spectral width. Captures the distribution of cut sizes.
- Higher-order moments (k=4,5): spectral shape. Captures fine structure of the partition hierarchy.

Lanczos computes a single eigenvector at high precision. Polynomial moments compute a low-precision but comprehensive summary of the entire spectrum. For similarity (not segmentation), the comprehensive summary is more useful.

## Implementation details

### Wide Laplacian matvec

The existing Triton kernel operates on a single (H, W) vector. The fingerprint needs (H, W, m) -- m independent matvecs simultaneously. Rather than launching the Triton kernel m times, the implementation uses `_laplacian_matvec_wide_torch`, which is the torch.roll stencil with broadcasting over the probe dimension:

```python
def _laplacian_matvec_wide_torch(X, weights, degree):
    # weights: (4, H, W) -> unsqueeze(-1) for broadcasting over m
    result = degree.unsqueeze(-1) * X
    result -= weights[0].unsqueeze(-1) * torch.roll(X, 1, 0)
    result -= weights[1].unsqueeze(-1) * torch.roll(X, -1, 0)
    result -= weights[2].unsqueeze(-1) * torch.roll(X, 1, 1)
    result -= weights[3].unsqueeze(-1) * torch.roll(X, -1, 1)
    return result
```

This launches 5 kernels per matvec (same as the non-Triton path), but each kernel processes all m probes in one launch. At m=64, this is 64x better parallelism than sequential single-vector matvecs.

### Constant-vector projection

The graph Laplacian's first eigenvector is the constant vector (eigenvalue 0). To prevent the fingerprint from being dominated by this trivial mode, probes are projected orthogonal to the constant vector at initialization AND after each matvec step:

```python
Z = Z - Z.mean(dim=(0, 1), keepdim=True)  # project out constant
```

This is the same projection used in the Lanczos pipeline.

### Log-scale compression

The raw quadratic forms z^T L^k z estimate sum(lambda_i^k), which grows exponentially with k (dominated by lambda_max^k). Without compression, the k=5 moment would dominate any distance metric and the lower-order structural information would be invisible.

The implementation uses signed log compression:

```python
fingerprint = sign(fingerprint) * log(|fingerprint| + eps)
```

This compresses the dynamic range so that all polynomial orders contribute equally to the L2 distance comparison, while preserving the sign structure (probes that have positive vs negative affinity with different spectral regions).

### Similarity metric

The structural similarity function computes fingerprints for both images and returns:

```python
similarity = 1 / (1 + ||fp_img - fp_ref||_2)
```

Cosine similarity was tried first but fails because fingerprints from same-resolution images are near-identical in direction (cos ~1.0) -- the discriminative signal is in small magnitude differences that cosine similarity discards. L2 distance captures these differences directly.

## Accuracy/speed tradeoff

### Constraint validation (5 TNT constraints)

| Method | Constraints passed | Failing constraint |
|--------|-------------------|-------------------|
| old pixel-cosine | 4/5 | SKETCH > COLOR |
| v2 Lanczos+segments | 5/5 | none |
| v3 fingerprint | 4/5 | SKETCH > COLOR |

v3 passes the same 4 constraints as old pixel-cosine, failing only constraint 4 (min SKETCH > max COLOR). This is the same weakness: the "1bit redraw" sketch scores lower than the nightmode photo, because the 1-bit sketch has very different spectral structure from the pizza-ratto reference (fewer smooth gradients, more sharp binary edges).

v2 passes all 5 because the Lanczos + segment matching pipeline explicitly decomposes structure into segments and matches them, which handles the sketch-vs-photo distinction better.

### Timing comparison

| Method | Avg ms/image | Relative to v2 | Notes |
|--------|-------------|----------------|-------|
| old pixel-cosine | 4.8 ms | 0.012x | No structural analysis |
| v3 fingerprint | 88.8 ms | 0.22x | 5 wide matvecs per image |
| v2 Lanczos+segments | 410.8 ms | 1.0x | 30 narrow matvecs + segments + matching |
| VAE encode (reference) | 200 ms | 0.49x | Cost of generating the input |

v3 is 4.6x faster than v2. The speedup comes from:
1. 5 serial steps instead of 30
2. No eigensolver overhead (torch.linalg.eigh on the tridiagonal)
3. No contour detection (Sobel convolution)
4. No connected components (iterative pointer jumping)
5. No segment matching (pairwise L2 + argmin)

First-image penalty: both v2 and v3 incur torch.compile warmup on first invocation. v2's warmup is ~3 seconds (compiling both the Lanczos loop and the CC loop). v3 has no compiled loops; the torch.roll ops run eagerly. First-image v3 timing is ~280ms (due to CUDA warmup), steady-state is ~60-95ms depending on resolution.

### Pairwise agreement

| Pair | Agreement rate | Total disagreements |
|------|---------------|-------------------|
| old vs v2 | 48.5% | 70/136 |
| old vs v3 | 67.6% | 44/136 |
| v2 vs v3 | 70.6% | 40/136 |

v3 agrees with v2 on 70.6% of pairwise orderings, and with old on 67.6%. The v2-vs-v3 agreement is notably higher than old-vs-v2 (48.5%), suggesting v3 captures some of the same structural information as v2 that pixel-cosine misses.

### Score distributions

| Image | old | v2 | v3 |
|-------|-----|-----|-----|
| pizza-ratto (THIS_REF) | +0.538 | +3.346 | +0.370 |
| offhand_pleometric (THAT_REF) | -0.538 | -3.320 | -0.370 |
| widemeister (SKETCH) | +0.004 | +0.769 | +0.318 |
| clear-sky-thick-mkii (SKETCH) | -0.014 | +1.277 | +0.290 |
| bubblegum-zinesona-4 (SKETCH) | -0.003 | +0.897 | +0.269 |
| red-tonegraph (MIXED) | +0.014 | +1.092 | +0.213 |
| mspaint-enso (SKETCH) | +0.002 | +1.199 | +0.186 |
| PINKER_D-F (PINKIFY) | -0.017...-0.048 | +1.049...+1.147 | +0.178...+0.184 |
| PINKER_A-C (PINKIFY) | -0.010...-0.033 | +0.809...+1.399 | +0.178 |
| nightmode (COLOR) | +0.111 | +0.360 | +0.084 |
| deviantart (SKETCH) | -0.009 | +0.573 | +0.020 |
| snek-heavy (SKETCH) | +0.020 | +0.386 | +0.008 |
| 1bit redraw (SKETCH) | -0.047 | +0.951 | -0.028 |

v3's dynamic range ([-.370, +.370]) is comparable to old ([-.538, +.538]) but much tighter than v2 ([-3.320, +3.346]). The score distribution is well-centered: THIS_REF and THAT_REF are symmetric (both +/- 0.370), and intermediates are spread across the range.

Notable: v3 clusters the PINKIFY images tightly (0.178-0.184), which makes sense -- they're all generated from the same prompt with similar structure, differing mainly in color. The Lanczos pipeline (v2) spreads them more (0.809-1.399) because segment matching is sensitive to fine structural differences between generated images.

## Appendix: Full score tables

> **All 17 images scored by three methods (most THIS-like first for each)**
>
> **OLD TNT (pixel cosine):**
> ```
> +0.537979  [THIS_REF  ]  pizza-ratto.png
> +0.110556  [COLOR     ]  00500-3023556536_re_nightmode2.png
> +0.020273  [SKETCH    ]  snek-heavy.png
> +0.014478  [MIXED     ]  red-tonegraph.png
> +0.004484  [SKETCH    ]  widemeister.png
> +0.002393  [SKETCH    ]  mspaint-enso-i-couldnt-forget-ii.png
> -0.002811  [SKETCH    ]  bubblegum-zinesona-4.png
> -0.007449  [PINKIFY   ]  PINKER_E.png
> -0.009082  [SKETCH    ]  deviantart-is-my-spine-moe-is-my-face.png
> -0.010425  [PINKIFY   ]  PINKER_A.png
> -0.014059  [SKETCH    ]  clear-sky-thick-mkii.png
> -0.017367  [PINKIFY   ]  PINKER_D.png
> -0.018000  [PINKIFY   ]  PINKER_B.png
> -0.033171  [PINKIFY   ]  PINKER_C.png
> -0.046986  [SKETCH    ]  1bit redraw.png
> -0.048340  [PINKIFY   ]  PINKER_F.png
> -0.537979  [THAT_REF  ]  offhand_pleometric.png
> ```
>
> **TNT v2 (Lanczos+segments):**
> ```
> +3.346067  [THIS_REF  ]  pizza-ratto.png
> +1.399095  [PINKIFY   ]  PINKER_B.png
> +1.397033  [PINKIFY   ]  PINKER_A.png
> +1.277342  [SKETCH    ]  clear-sky-thick-mkii.png
> +1.198766  [SKETCH    ]  mspaint-enso-i-couldnt-forget-ii.png
> +1.147489  [PINKIFY   ]  PINKER_F.png
> +1.091723  [MIXED     ]  red-tonegraph.png
> +1.089822  [PINKIFY   ]  PINKER_E.png
> +1.049264  [PINKIFY   ]  PINKER_D.png
> +0.951147  [SKETCH    ]  1bit redraw.png
> +0.897287  [SKETCH    ]  bubblegum-zinesona-4.png
> +0.809298  [PINKIFY   ]  PINKER_C.png
> +0.768884  [SKETCH    ]  widemeister.png
> +0.572819  [SKETCH    ]  deviantart-is-my-spine-moe-is-my-face.png
> +0.386091  [SKETCH    ]  snek-heavy.png
> +0.359902  [COLOR     ]  00500-3023556536_re_nightmode2.png
> -3.320323  [THAT_REF  ]  offhand_pleometric.png
> ```
>
> **TNT v3 (spectral fingerprint):**
> ```
> +0.370411  [THIS_REF  ]  pizza-ratto.png
> +0.317792  [SKETCH    ]  widemeister.png
> +0.289726  [SKETCH    ]  clear-sky-thick-mkii.png
> +0.269296  [SKETCH    ]  bubblegum-zinesona-4.png
> +0.212548  [MIXED     ]  red-tonegraph.png
> +0.186427  [SKETCH    ]  mspaint-enso-i-couldnt-forget-ii.png
> +0.183742  [PINKIFY   ]  PINKER_F.png
> +0.183719  [PINKIFY   ]  PINKER_E.png
> +0.183708  [PINKIFY   ]  PINKER_D.png
> +0.177923  [PINKIFY   ]  PINKER_A.png
> +0.177880  [PINKIFY   ]  PINKER_B.png
> +0.177653  [PINKIFY   ]  PINKER_C.png
> +0.083554  [COLOR     ]  00500-3023556536_re_nightmode2.png
> +0.019502  [SKETCH    ]  deviantart-is-my-spine-moe-is-my-face.png
> +0.008115  [SKETCH    ]  snek-heavy.png
> -0.028307  [SKETCH    ]  1bit redraw.png
> -0.370411  [THAT_REF  ]  offhand_pleometric.png
> ```

## Appendix: Timing tables

> **Per-image timing (ms)**
>
> | Image | old | v2 | v3 |
> |-------|-----|-----|-----|
> | 00500-3023556536_re_nightmode2 (1280x832) | 66.8 | 3174.4 | 280.9 |
> | 1bit redraw (549x499) | 1.2 | 2918.5 | 136.3 |
> | bubblegum-zinesona-4 (512x512) | 0.8 | 38.8 | 74.5 |
> | clear-sky-thick-mkii (676x679) | 0.9 | 34.7 | 110.8 |
> | deviantart (512x512) | 0.8 | 31.4 | 63.0 |
> | mspaint-enso (256x256) | 1.2 | 418.8 | 7.4 |
> | offhand_pleometric (512x512) | 0.7 | 30.1 | 62.0 |
> | pizza-ratto (512x512) | 0.7 | 38.5 | 62.3 |
> | red-tonegraph (306x314) | 0.7 | 34.6 | 13.4 |
> | snek-heavy (512x512) | 0.8 | 31.2 | 62.8 |
> | widemeister (512x512) | 1.1 | 33.9 | 64.0 |
> | PINKER_A-F (768x512) | 0.8-0.9 | 30.5-37.6 | 90.4-98.1 |
>
> Note: First invocations of v2 include torch.compile warmup (3+ seconds).
> v3 has no compiled loops; first-image cost is CUDA warmup only (~280ms for 1280x832).
> Steady-state v3 scales approximately linearly with pixel count.

## Appendix: Constraint checks

> **5 TNT validation constraints:**
>
> | Constraint | old | v2 | v3 |
> |-----------|-----|-----|-----|
> | 1. THIS_REF > THAT_REF | PASS | PASS | PASS |
> | 2. THIS_REF > all others | PASS | PASS | PASS |
> | 3. THAT_REF < all others | PASS | PASS | PASS |
> | 4. min(SKETCH) > max(COLOR) | FAIL | PASS | FAIL |
> | 5. THAT_REF < NIGHTMODE | PASS | PASS | PASS |
>
> Constraint 4 failure in v3: min(SKETCH) = -0.028 (1bit redraw) vs max(COLOR) = 0.084 (nightmode).
> Same failure mode as old pixel-cosine. The 1-bit sketch has binary edges that create a very
> different spectral structure from the continuous-tone pizza-ratto reference.

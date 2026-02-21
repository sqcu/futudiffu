# TNT v4: Zernike-Probed Spectral Fingerprinting

## 1. What this is

TNT (ThisNotThat) scoring compares a generated image against two reference
images -- THIS (pizza-ratto) and THAT (offhand_pleometric) -- returning a
signed score. Positive = more like THIS. Negative = more like THAT.

v4 replaces v3's random probe vectors with Zernike polynomial basis functions
to probe the graph Laplacian spectrum. The motivation: v3 uses random Gaussian
probes to compute stochastic spectral moments via quadratic forms z^T L^k z.
These probes have no rotational structure, so the fingerprint changes
unpredictably under fractional rotations. Zernike polynomials, by contrast,
are the canonical orthogonal basis on the unit disk and have well-defined
angular behavior: R_n^m(r) * exp(i*m*theta) transforms as exp(i*m*alpha)
under rotation by alpha. This should make the spectral fingerprint more
stable under rotations.

The pipeline:
1. Build color-aware edge weights (same as v2/v3)
2. Compute Zernike polynomial basis Z on the (H, W) pixel grid (44 probes for max_order=8)
3. Apply T=5 iterated Laplacian matvecs to the probe matrix: X <- L * X
4. At each step, collect quadratic forms: fp[t, k] = z_k^T * L^t * z_k
5. Apply signed log compression
6. Compare fingerprints via L2 distance -> similarity = 1 / (1 + dist)

## 2. Zernike radial polynomial implementation

Valid (n, m) pairs: n >= 0, |m| <= n, (n - |m|) even. For max_order=8, this
produces 44 probe channels (excluding the constant mode n=0, m=0). For each
mode with m > 0, we generate both cos(m*theta) and sin(m*theta) components as
separate probes, giving full angular information.

The radial polynomial:
```
R_n^m(r) = sum_{s=0}^{(n-|m|)/2} (-1)^s * (n-s)! / (s! * ((n+|m|)/2-s)! * ((n-|m|)/2-s)!) * r^{n-2s}
```

Pixels outside the inscribed unit disk (radius = min(H, W) / 2) get value 0.
All probes are projected orthogonal to the constant vector and normalized to
unit norm, matching v3's preprocessing.

## 3. Constraint results (Part A)

Scored 11 images in i2i_off_policies/ with all 4 methods.

| Constraint | v1 (pixel cos) | v2 (Lanczos) | v3 (random fp) | v4 (Zernike fp) |
|---|---|---|---|---|
| 1. THIS > THAT | PASS | PASS | PASS | PASS |
| 2. THIS > all others | PASS | PASS | PASS | PASS |
| 3. THAT < all others | PASS | PASS | PASS | PASS |
| 4. min(SKETCH) > max(COLOR) | FAIL | PASS | FAIL | FAIL |
| 5. THAT < NIGHTMODE | PASS | PASS | PASS | PASS |
| **Total** | **4/5** | **5/5** | **4/5** | **4/5** |

v4 passes the same 4 constraints as v1 and v3. Only v2 passes all 5, because
its segment-matching approach creates a much larger dynamic range between
the THIS/THAT refs and everything else.

v4's failure on constraint 4 is structural: the Zernike spectral fingerprint
doesn't distinguish well between sketch-style images and photographic images
when both are dissimilar to the THIS reference. The scores for non-reference
images cluster in a narrow band around -0.10 to -0.13, with poor tier separation.

## 4. Timing comparison

| Method | Avg ms/image |
|---|---|
| v1 (pixel cosine) | 7.1 |
| v2 (Lanczos+segments) | 651.3 |
| v3 (random fingerprint) | 86.7 |
| v4 (Zernike fingerprint) | 71.5 |

v4 is slightly faster than v3 (71.5 vs 86.7 ms) because the Zernike basis is
precomputed once per resolution, while v3 generates 64 random probes each time.
Both are comfortably within the ~200ms VAE encode budget. v2 remains the slowest
due to its serial Lanczos eigendecomposition.

## 5. Invariance results (Part B)

16 transforms tested: 10 basic + 6 fractional rotations. Two anchors (pizza-ratto
and offhand_pleometric). Pass = sign preserved after transform.

### 5.1 Overall pass rates

| Method | pizza-ratto | offhand_pleometric | Overall |
|---|---|---|---|
| v1 (pixel cosine) | 94% (15/16) | 88% (14/16) | 29/32 (91%) |
| v2 (Lanczos) | 100% (16/16) | 50% (8/16) | 24/32 (75%) |
| v3 (random fp) | 38% (6/16) | 100% (16/16) | 22/32 (69%) |
| v4 (Zernike fp) | 75% (12/16) | 62% (10/16) | 22/32 (69%) |

### 5.2 Basic transform results

| Transform | v1 | v2 | v3 | v4 |
|---|---|---|---|---|
| identity | OK/OK | OK/OK | OK/OK | OK/OK |
| horizontal_flip | OK/OK | OK/OK | OK/OK | OK/OK |
| rotate_90 | OK/OK | OK/OK | OK/OK | OK/OK |
| shear (0.15) | OK/OK | OK/FAIL | FAIL/OK | OK/FAIL |
| scale (70%+pad) | OK/FAIL | OK/FAIL | OK/OK | FAIL/OK |
| hue shift (+60 deg) | OK/OK | OK/OK | OK/OK | OK/OK |
| color inversion | FAIL/FAIL | OK/OK | OK/OK | OK/OK |
| Gaussian blur (sigma=3) | OK/OK | OK/OK | FAIL/OK | OK/FAIL |
| Gaussian noise (sigma=0.05) | OK/OK | OK/OK | FAIL/OK | OK/FAIL |
| center crop (80%) | OK/OK | OK/OK | FAIL/OK | OK/FAIL |

(Format: pizza-ratto/offhand_pleometric)

Key observations:
- v4 is **perfectly invariant to hue shift and color inversion** (same as v3), because the edge weights are based on RGB L2 distance, which is invariant to uniform color transforms.
- v4 fails under Gaussian blur (sigma=3) for offhand_pleometric and under Gaussian noise for offhand_pleometric. These destroy high-frequency graph structure that the Zernike probes are sensitive to.
- v4 fails for scale+pad on pizza-ratto, likely because the zero-padded border creates a strong disk-boundary artifact in the Zernike basis.

### 5.3 Fractional rotation results (the key question)

| Rotation | v1 | v2 | v3 | v4 |
|---|---|---|---|---|
| 72 deg (pentagonal) | OK/OK | OK/FAIL | FAIL/OK | OK/OK |
| 51.4 deg (septagonal) | OK/OK | OK/FAIL | FAIL/OK | FAIL/OK |
| 40 deg (nonagonal) | OK/OK | OK/FAIL | FAIL/OK | FAIL/OK |
| 15 deg | OK/OK | OK/FAIL | FAIL/OK | OK/FAIL |
| 7 deg | OK/OK | OK/FAIL | FAIL/OK | OK/FAIL |
| 137.5 deg (golden) | OK/OK | OK/FAIL | FAIL/OK | FAIL/OK |

(Format: pizza-ratto/offhand_pleometric)

Fractional rotation pass rates:
- **v1: 12/12** (100%) -- pixel cosine is surprisingly robust
- **v2: 6/12** (50%) -- all failures on offhand_pleometric
- **v3: 6/12** (50%) -- all failures on pizza-ratto
- **v4: 7/12** (58%) -- mixed failures on both anchors

v4 improves marginally over v3 on fractional rotations (7/12 vs 6/12), but the
improvement is modest. Three of v4's five fractional rotation failures are on
pizza-ratto (septagonal, nonagonal, golden angle), and two are on offhand_pleometric
(15 deg, 7 deg).

The key insight: **v3 and v4 fail on opposite anchors**. v3 fails all 6
rotations on pizza-ratto but passes all 6 on offhand_pleometric. v4 passes
3/6 on pizza-ratto and 4/6 on offhand_pleometric. The Zernike probes shift
the failure distribution but don't eliminate it.

## 6. Why v4 does not fully solve rotation invariance

The theoretical argument for Zernike rotation invariance assumes the probes
operate on a continuous unit disk where rotation commutes with the angular
basis functions. On a discrete pixel grid with image-dependent edge weights:

1. **Discretization breaks symmetry.** The Zernike basis is computed on a
   fixed pixel grid. After rotation via affine_grid + grid_sample with
   bilinear interpolation, the pixels sample different positions. The
   Zernike basis vectors are not recomputed for the rotated grid -- they
   remain fixed. So the quadratic forms z^T L^k z use the same probes
   on a different graph, rather than rotating the probes to match.

2. **Image-dependent edge weights.** The Laplacian is built from RGB
   color distances. Rotating the image changes which pixel pairs are
   neighbors in the 4-connected grid, and bilinear interpolation blurs
   the color boundaries. The rotated image has a different graph structure,
   not just a rotated version of the same graph.

3. **Boundary effects.** Reflection padding at the rotation boundary
   introduces non-physical image content. The Zernike basis's unit disk
   mask partially mitigates this (pixels outside the inscribed circle get
   zero probe weight), but reflection artifacts inside the circle still
   affect the edge weights and hence the Laplacian.

4. **The fundamental mismatch.** True rotation invariance would require
   either (a) rotating the Zernike probes along with the image, or (b)
   using rotation-invariant features of the fingerprint (e.g., the
   magnitude of the complex Zernike moments, discarding phase). v4 does
   neither -- it uses the raw quadratic forms which contain phase
   information that changes under rotation.

## 7. False-color visualization discussion

All visualizations saved to `tnt_v4_zernike_validation/visualizations/`.

### 7.1 Fiedler vector field

The Fiedler vector (2nd smallest eigenvector of the graph Laplacian) shows
the primary spectral partition of each image. For pizza-ratto, the Fiedler
field clearly separates the subject from the background. For offhand_pleometric,
it captures the dominant geometric structure. Under 72-degree rotation, the
Fiedler field rotates with the image content but is not identical to the
original rotated -- the discretization and reflection padding introduce
visible artifacts, particularly at boundaries.

### 7.2 Random probe fields (v3)

The random probe fields L^k z for k=1,3,5 show increasingly fine-grained
spectral structure as k increases. At L^1, the fields capture low-frequency
graph structure. By L^5, the fields contain high-frequency oscillations that
are sensitive to pixel-level graph connectivity -- this is why v3 breaks
under blur and noise.

### 7.3 Zernike basis functions

The first 12 Zernike basis functions display the expected spatial structure:
- Mode 0 (n=1, m=1, cos): horizontal gradient
- Mode 1 (n=1, m=1, sin): vertical gradient
- Mode 2 (n=2, m=0): radial "defocus" ring
- Higher modes: increasingly complex angular and radial oscillations
- The unit disk mask is clearly visible as a black border outside the
  inscribed circle

### 7.4 Zernike-probed spectral fields

Compared to the random probe fields, the Zernike-probed fields L^k z_k
show more structured spatial patterns that reflect the angular order of
the Zernike mode. The low-order modes (n=1, m=1) produce broad fields
similar to the Fiedler vector, while higher-order modes produce finer
angular structure.

### 7.5 Rotation comparison

Side-by-side comparison of original and 72-degree-rotated fields confirms
the visual impression: the Fiedler field rotates approximately with the
image but is not identical. The Zernike-probed fields show more structural
similarity between original and rotated versions than the random probe fields,
but the match is imperfect due to the discrete grid effects described above.

## 8. Assessment

**Does v4 solve the rotation invariance problem?** No. v4 achieves 7/12 on
fractional rotations, a marginal improvement over v3's 6/12 but far from
v1's 12/12 or the ideal 12/12.

**What v4 does achieve:**
- Perfect invariance to hue shift and color inversion (inherited from v3's edge weight construction)
- Slightly better rotation behavior than v3 (shifts failures from all-one-anchor to distributed)
- Comparable timing to v3 (~72ms vs ~87ms)
- Same constraint pass rate as v1 and v3 (4/5)

**What would actually solve rotation invariance:**
1. **Rotate the probes.** Instead of using fixed Zernike basis on the pixel grid,
   compute rotation-invariant features: use |z^T L^k z| (magnitude) instead of
   the raw quadratic form. This discards phase but preserves the spectral content.
2. **Use rotationally-invariant graph features.** The eigenvalue spectrum of the
   Laplacian is rotation-invariant. Spectral moments tr(L^k) are invariant.
   The problem is that these are global features that lack image discrimination.
3. **Ensemble over rotations.** Compute fingerprints at multiple rotation angles
   and take the min-distance match. Expensive (T*K times more matvecs) but correct.
4. **Polar coordinate graph.** Build the Laplacian on a polar grid rather than
   a Cartesian pixel grid. Rotation becomes a shift in the angular coordinate,
   which is handled by the circulant structure of the polar Laplacian.

The simplest viable fix for a v5 would be option 1: using |quadratic form|
magnitude instead of the signed value. This sacrifices some discriminative
power (losing sign information) but should dramatically improve rotation
invariance by eliminating the phase dependence that causes the failures.

## 9. Data files

All intermediate tensors and scores saved to `tnt_v4_zernike_validation/`:
- `part_a_scores.json` -- all image scores and constraint results
- `part_b_invariance.json` -- all transform scores and pass/fail
- `summary.json` -- aggregate pass rates and timing
- `visualizations/` -- Fiedler fields, random probe fields, Zernike basis,
  Zernike-probed fields, rotation comparison panels, saved `.pt` tensors

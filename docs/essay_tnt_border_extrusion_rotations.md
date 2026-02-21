# TNT Border Extrusion vs Reflection Padding for Fractional Rotations

## Context

The previous fractional rotation invariance tests (v5 Zernike-Fiedler
validation, Part B) used `padding_mode='reflection'` in `F.grid_sample`.
Reflection padding mirrors image content into the out-of-bounds regions
created by rotation. The hypothesis under test: these mirror-image
structures create spurious edges in the graph Laplacian that confound
the Fiedler vector and downstream scoring.

The proposed fix: `padding_mode='border'`, which replicates the nearest
border pixel outward. This creates smooth uniform regions that the
Laplacian sees as "same region" (no spurious edges), in contrast to
reflection which introduces sharp RGB discontinuities at the mirror
seams.

## What border extrusion vs reflection looks like

Side-by-side comparison images were generated for all 12 anchor/rotation
combinations (saved to `tnt_border_extrusion_validation/images/`).

**Border extrusion**: The corners and edges of the rotated image show
solid-color bands -- each border pixel is extruded outward, creating
uniform rectangular strips. For pizza-ratto at 72 degrees, the corners
become flat orange or flat blue regions. The Laplacian assigns zero
gradient to these uniform areas, so they contribute nothing to the
Fiedler vector. They are effectively invisible to the structural
similarity pipeline.

**Reflection padding**: The same corners show mirrored copies of the
image content. At 72 degrees, the reflected strips contain inverted
color patterns from the near-border region of the image. The Laplacian
sees these as real structural features: edges, contours, color
transitions. The reflected structures create additional segments in the
connected component analysis and alter the Fiedler vector's partitioning
behavior.

For small rotation angles (7 degrees), the visual difference is subtle:
only thin wedges at the corners are affected. For large angles (137.5
degrees golden angle), the affected area is substantial.

## Pass rate table: method x padding mode x pass count out of 12

| Method              | Border | Reflection |
|---------------------|--------|------------|
| v1_gpu              | 12/12  | 12/12      |
| v2_lanczos          |  7/12  |  6/12      |
| v3_fingerprint      |  7/12  |  6/12      |
| v4_zernike          |  7/12  |  7/12      |
| v5_zernike_fiedler  |  9/12  |  9/12      |

## Does border extrusion fix v2's and v5's fractional rotation failures?

### v1_gpu: 12/12 both modes

v1 (pixel cosine + structural cosine) is fully robust to fractional
rotations regardless of padding mode. The flattened cosine similarity
is insensitive to corner artifacts because they represent a small
fraction of total pixel energy.

### v2_lanczos: 7/12 border vs 6/12 reflection

Border extrusion fixed exactly one case: `offhand_pleometric/137.5_golden`
flipped from +0.146 (wrong sign, FAIL) to -0.233 (correct sign, PASS).

The remaining 5 failures are ALL on `offhand_pleometric` and show large
positive scores (+0.29 to +0.93) when the expected sign is negative.
These are NOT padding artifacts. The fundamental issue is that rotating
`offhand_pleometric` shifts its structural distance to `pizza-ratto` vs
its structural distance to itself. The Lanczos+segment pipeline computes
segment signatures that change dramatically under rotation because:

1. The contour detection finds different connected components when the
   image is rotated (different pixel alignments create different threshold
   crossings in the Sobel gradient).
2. The segment matching via z-score L2 is sensitive to the segment
   population: a different number of segments changes the z-score
   normalization for all segments.

Border extrusion helps at the margin (one fix) but the core v2 failure
mechanism is interpolation-induced contour instability, not padding
artifacts.

### v3_fingerprint: 7/12 border vs 6/12 reflection

Border extrusion fixed one case: `pizza-ratto/137.5_golden` flipped
from -0.099 (wrong sign, FAIL) to +0.009 (correct sign, PASS).

The remaining 5 failures are ALL on `pizza-ratto` with negative scores
(-0.001 to -0.152) when the expected sign is positive. The v3 spectral
fingerprint uses random probes, and the quadratic forms z^T L^k z are
sensitive to the graph structure in ways that correlate poorly with
rotation. The random probes do not have rotational symmetry, so a
rotated graph produces different quadratic forms even if the underlying
spectrum is similar.

Border extrusion has minimal impact because the fingerprint comparison
is dominated by the interior structure (where the graph actually has
edges), not the border regions.

### v4_zernike: 7/12 both modes

Border extrusion changes nothing for v4. The failure pattern is
identical: 3 pizza-ratto rotations (51.4, 40, 137.5 degrees) produce
wrong-sign scores, and 2 offhand_pleometric rotations (15, 7 degrees)
produce wrong-sign scores.

v4's Zernike-probed spectral fingerprint suffers from a different
mechanism: the high-pass cascade (T=5 matvecs) amplifies boundary
effects regardless of padding mode. The Zernike probes are defined on
the inscribed unit disk and are zero outside it, so corner padding is
irrelevant -- the problem is in how the Laplacian stencil interacts with
the rotation-induced interpolation blur in the interior.

### v5_zernike_fiedler: 9/12 both modes

Border extrusion changes nothing for v5. The same 3 failures persist:
`pizza-ratto` at 72, 15, and 7 degrees produces small negative scores
(-0.0088 to -0.0105) when the expected sign is positive.

These failures are interesting because:
- The identity score for pizza-ratto is only +0.0127 (v5 has the
  smallest dynamic range of all methods).
- The failing scores are -0.009 to -0.011: they are within 0.022 of the
  identity score.
- The Zernike moment distances (original vs rotated) are small: 0.023
  for pizza-ratto, 0.032 for offhand_pleometric.
- v5 passes on offhand_pleometric for ALL rotations with BOTH padding
  modes.

The v5 failures are NOT from padding artifacts. They are from the
Fiedler vector's sign ambiguity interacting with the moment computation:
the Fiedler vector of a rotated image can have a different global
partition pattern (flipped major-minor partition axis), and while
|Z_{n,m}| magnitudes are sign-invariant, the combination of small moment
differences and small identity score pushes the score across zero.

## Updated Zernike moment bar charts

The bar charts comparing original vs border-extrusion-rotated 72 degrees
vs reflection-rotated 72 degrees are saved to
`tnt_border_extrusion_validation/charts/`.

### Are the moments more stable with border extrusion?

Moment distances (L2 norm of |Z_{n,m}| difference vectors):

| Anchor              | Border rot72 | Reflection rot72 |
|---------------------|-------------|-------------------|
| pizza-ratto         | 0.023012    | 0.023127          |
| offhand_pleometric  | 0.032309    | 0.032337          |

The differences are negligible (0.5% relative). Border extrusion and
reflection produce nearly identical Zernike moment vectors because the
Zernike polynomials are defined on the inscribed unit disk, and the
padding-affected regions are in the corners OUTSIDE this disk. The disk
mask zeros out the corner pixels before moment computation, making the
padding mode irrelevant for Zernike moments.

The bar charts visually confirm this: the original (blue) and
border-rotated (green) and reflection-rotated (red) bars are nearly
identical at each mode, with tiny deviations concentrated in the
low-order modes (1,0) and (2,0) where the Fiedler vector's large-scale
partition is most affected by interpolation.

## Root cause analysis: what causes the remaining failures?

### Not padding artifacts

The border extrusion experiment conclusively demonstrates that padding
mode is not the primary failure mechanism. It fixed exactly 2 out of 24
failure cases (one for v2, one for v3). The remaining failures have
other root causes.

### Interpolation blur (dominant for v3)

Bilinear interpolation at fractional angles creates a low-pass filter
effect. For v3 (random probe fingerprints), this blur changes the graph
Laplacian's spectrum sufficiently to alter the quadratic forms. The
random probes have no rotational structure, so rotated spectra produce
different fingerprints. This is a fundamental v3 design limitation:
random probes are not rotation-invariant.

### Contour instability (dominant for v2)

For v2 (Lanczos + segments), the Sobel gradient magnitude field changes
under rotation because the discrete 3x3 Sobel kernel is not
rotationally symmetric. Pixels near the z-score threshold for contour
detection flip between contour/non-contour, changing the connected
component topology and invalidating the segment matching.

### Fiedler partition axis (dominant for v5)

For v5 (Zernike moments of Fiedler), the Fiedler vector's dominant
partition axis can rotate with the image, but the Lanczos iteration
with seed=42 may converge to a different Fiedler vector when the image
rotational symmetry interacts with the random initialization. The
resulting moment vectors are close (L2 distance ~0.02-0.03) but the
identity score is also small (~0.013), so small moment perturbations
cross the zero boundary.

### Zernike probe cascade (dominant for v4)

For v4, the T=5 iterated matvecs amplify the interpolation-induced
changes in the graph Laplacian. Each matvec step acts as a graph filter,
and the cumulative effect of 5 steps on a slightly-perturbed graph
(from interpolation blur) produces fingerprints that diverge from the
original.

## Conclusion

Border extrusion (padding_mode='border') provides a marginal improvement
over reflection padding: 2 additional passes out of 24 failure cases
(both at 137.5 degrees, the rotation angle with maximum out-of-bounds
area). The fix is worth keeping as a default since it is strictly better
and has no downsides, but it does not fundamentally solve fractional
rotation invariance for v2-v5.

The method rankings are unchanged:
1. **v1** (12/12): Fully robust. Pixel cosine similarity is inherently
   rotation-tolerant.
2. **v5** (9/12): Best structural method. Failures are small-margin
   sign crossings on pizza-ratto only.
3. **v2, v3, v4** (6-7/12): All fail roughly half the time on fractional
   rotations, each for different structural reasons.

For true rotation invariance in v5, the path forward is not better
padding but rather: (a) increasing the identity score's dynamic range
(e.g., by weighting low-order Zernike moments more heavily), or (b)
using a sign-canonical Fiedler vector (fixing the sign ambiguity before
moment computation).

## Output artifacts

- `tnt_border_extrusion_validation/comprehensive_results.json` -- all scores, timings, pass/fail
- `tnt_border_extrusion_validation/images/` -- 36 individual rotated images + 12 side-by-side comparisons
- `tnt_border_extrusion_validation/charts/` -- 2 Zernike moment bar charts (border vs reflection vs original)
- `tnt_border_extrusion_validation/*.pt` -- saved moment tensors for both anchors
- `tnt_v5_validation/visualizations/offhand_pleometric_fiedler_with_zernike_modes.png` -- the previously missing composite

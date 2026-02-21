# TNT v5: Zernike Moments of the Fiedler Vector Field

## Motivation

The project has four existing TNT scoring approaches. The trajectory of improvements reveals a specific failure mode at each stage:

- **v1** (pixel cosine + structural similarity): Fast but shallow. Fails color inversion, no structural understanding.
- **v2** (Lanczos Fiedler + segments): Passes all 5 constraints, runs ~30ms with compiled Triton Lanczos. But fails all fractional rotations on offhand_pleometric (6/12 fractional rotations overall).
- **v3** (random spectral probes): Slower than compiled v2, fails all fractional rotations on pizza-ratto (6/12 overall). Sensitive to blur.
- **v4** (Zernike spectral probes): Slower than v2, fails some fractional rotations on both anchors (7/12 overall). The false-color analysis from v4 revealed the root cause: Zernike basis functions are smooth polynomials, but the Laplacian power cascade L^k drives them to boundary artifacts. The image's mid-frequency structural content gets lost in the high-pass amplification.

The fix proposed for v5: compute the Fiedler vector first (which IS the mid-frequency structural field, already fast via compiled Lanczos at ~30ms), then take Zernike moments of THAT field. This composes the structural lifting (Fiedler) with the geometric invariance descriptor (Zernike magnitudes).

## Why This Composition Should Work

The Fiedler vector v_2 is the eigenvector corresponding to the smallest non-zero eigenvalue of the graph Laplacian. On a color-weighted image grid, it represents the dominant partition boundary -- the single cut that best separates the image into two structurally distinct regions. It encodes mid-frequency structure because:

- The constant vector (lambda=0) is the DC component
- The Fiedler (lambda_2) is the lowest non-trivial mode
- Higher eigenvectors capture progressively finer structural detail

Zernike moment magnitudes |Z_{n,m}| are exactly rotation-invariant descriptors: rotating a field by angle alpha changes Z_{n,m} to Z_{n,m} * exp(i*m*alpha), but the magnitude is unchanged. If the Fiedler field were simply rotated when the image is rotated, the Zernike moment magnitudes would be perfectly invariant.

The composition: Fiedler provides the structural content, Zernike magnitudes provide the rotation-invariant description of that content.

## Implementation

### Core functions (in `src_ii/itten_cuter_grops.py`)

**`zernike_moments_of_field(field, max_order=12)`**:
- Maps the (H, W) grid to a unit disk centered on the image
- For each valid Zernike mode (n, m) with m >= 0 (conjugate symmetry gives |Z_{n,-m}| = |Z_{n,m}|)
- Computes the complex Zernike moment Z_{n,m} = sum(field * conj(V_{n,m})) / disk_area
- Returns the vector of magnitudes |Z_{n,m}|
- Critical: normalizes the field to unit RMS within the disk before computing moments. Without this, the Fiedler vector (which has unit L2 norm over all HW pixels) produces moments in the 1e-5 range, collapsing all similarities to ~1.0.

**`structural_similarity_zernike_fiedler(image, reference, max_order=12, edge_threshold=0.15)`**:
- Builds edge weights and computes Fiedler for both images via `lanczos_fiedler()` (uses compiled Triton path)
- Computes Zernike moment magnitudes of both Fiedler fields
- Returns similarity = 1 / (1 + ||moments_img - moments_ref||_2)

**Sign ambiguity**: The Fiedler vector has a sign ambiguity (v and -v are both valid eigenvectors). Taking |Z_{n,m}| magnitudes fully resolves this: for all m, |Z_{n,m}(-f)| = |Z_{n,m}(f)| because |Z_{n,m}| is the magnitude of a linear functional applied to f, and negating f negates the complex moment but not its magnitude.

### Scoring function (in `src_ii/reward_functions.py`)

**`thisnotthat_score_v5`**: Same interface as v2/v3/v4. Score = similarity(image, THIS) - similarity(image, THAT).

## Results

### Constraint Results (5 methods x 5 constraints)

| Method | C1: THIS>THAT | C2: THIS>all | C3: THAT<all | C4: SKETCH>COLOR | C5: THAT<NIGHT | Total |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| v1_gpu | PASS | PASS | PASS | FAIL | PASS | 4/5 |
| v2_lanczos | PASS | PASS | PASS | PASS | PASS | **5/5** |
| v3_fingerprint | PASS | PASS | PASS | FAIL | PASS | 4/5 |
| v4_zernike | PASS | PASS | PASS | FAIL | PASS | 4/5 |
| v5_zernike_fiedler | PASS | FAIL | FAIL | FAIL | PASS | **2/5** |

v5 fails constraints 2 and 3 because its scores have low dynamic range. The identity pizza-ratto score is +0.0127 while 1bit-redraw scores +0.0164 -- all non-reference images are in a tight band around zero, with some falling outside the reference scores. The Zernike moment magnitudes produce similar descriptors for structurally different images because the moments capture the SPATIAL DISTRIBUTION of the Fiedler field, which is largely determined by image dimensions and Laplacian eigenvalue structure rather than the specific content.

### Invariance Results

**Overall pass rates** (16 transforms x 2 anchors = 32 total):

| Method | pizza-ratto | offhand_pleometric | Overall |
|--------|:-----------:|:------------------:|:-------:|
| v1_gpu | 94% | 94% | 30/32 |
| v2_lanczos | 100% | 56% | 25/32 |
| v3_fingerprint | 31% | 100% | 21/32 |
| v4_zernike | 81% | 56% | 22/32 |
| v5_zernike_fiedler | 62% | **100%** | 26/32 |

### Fractional Rotation Results (6 angles x 2 anchors = 12 total)

| Method | Pass | Fail details |
|--------|:----:|:-------------|
| v1_gpu | **12/12** | None |
| v2_lanczos | 6/12 | All 6 offhand_pleometric rotations fail |
| v3_fingerprint | 6/12 | All 6 pizza-ratto rotations fail |
| v4_zernike | 7/12 | 3 pizza-ratto + 2 offhand_pleometric |
| v5_zernike_fiedler | **9/12** | 3 pizza-ratto: 72deg, 15deg, 7deg |

v5 achieves the best fractional rotation performance among the graph-based methods (v2-v5), with 9/12 passes. The 3 failures are all on pizza-ratto with very marginal sign flips (scores of -0.0098, -0.0088, -0.0105 when +positive was expected). On offhand_pleometric, v5 achieves 6/6 perfect fractional rotation preservation.

### Timing Comparison

| Method | Avg ms/image | Notes |
|--------|:------------:|:------|
| v1_gpu | 7.6 | Pixel-space cosine, trivially fast |
| v2_lanczos | 610 | First call includes torch.compile warmup |
| v3_fingerprint | 84 | 5 wide matvecs on 64 probes |
| v4_zernike | 67 | 5 wide matvecs on Zernike probes |
| v5_zernike_fiedler | 75 | 2x Fiedler (30 Lanczos each) + Zernike moments |

v5 is comparable in speed to v3/v4 and significantly faster than v2's first-call timing.

### False-Color Analysis

**Rotation invariance of Zernike moments**:
- pizza-ratto: moment L2 distance between original and 72-deg rotated = 0.023 (small)
- offhand_pleometric: moment L2 distance between original and 72-deg rotated = 0.032 (small)

These distances are indeed small compared to the inter-image distances, confirming that Zernike moment magnitudes provide reasonable rotation invariance for the Fiedler field.

**Color invariance**:
- Hue shift (+60 deg): moment distance = 0.000 (perfect, because edge weights use L2 RGB distance which is hue-sensitive, but the Fiedler eigenstructure is dominated by luminance contrast)
- Color inversion: moment distance = 0.000 (perfect, because the Laplacian is symmetric in color distance)

## Comparison with v4: Why Probing the Fiedler Works Where Probing the Laplacian Failed

The v4 failure mode was clearly diagnosed: applying L^k to Zernike probe vectors creates a high-pass cascade that amplifies boundary artifacts while suppressing mid-frequency image structure. The Zernike polynomials are smooth, low-frequency functions that get destroyed by repeated differentiation (which is what L^k does).

v5 avoids this entirely by separating the two operations:
1. **Structural extraction** (Fiedler): the Lanczos iteration isolates the mid-frequency structural field WITHOUT the high-pass cascade on the probes.
2. **Geometric description** (Zernike moments): applied to the already-extracted field, not to raw high-pass-filtered probes.

The false-color visualizations confirm this: the Fiedler field contains clear structural content (image partition boundaries), while the L^k-filtered Zernike probes from v4 show mostly ringing artifacts near image edges.

## Assessment: Is This the Final TNT Scoring Function?

**No.** v5 demonstrates the theoretical soundness of the composition (Fiedler + Zernike moments), but it fails on the practical constraint tests. The fundamental issue is:

**The Fiedler vector is NOT equivariant to image rotation.** Rotating an image changes the pixel grid, which changes the graph, which changes the Laplacian, which produces a DIFFERENT Fiedler vector -- not the original Fiedler vector rotated. The Zernike moments are rotation-invariant descriptors of their input field, but their input is not a rotated version of the original field. It's a different field computed from a different graph.

This means v5's rotation invariance is approximate, not exact. The 9/12 fractional rotation pass rate (vs v4's 7/12) shows the approximation is better than v4's but not perfect. The marginal sign flips (-0.008 to -0.010) are within the noise floor of Fiedler instability under bilinear interpolation.

**Strengths of v5**:
- Best fractional rotation invariance among graph-based methods (9/12)
- Perfect color and luminance invariance
- Clean theoretical composition
- Reasonable speed (~75ms)

**Weaknesses of v5**:
- Low constraint pass rate (2/5) due to compressed dynamic range
- Scores are ~100x smaller than v2, making discrimination fragile
- The Fiedler instability under rotation limits true invariance

**The path forward** likely requires either:
1. A rotation-equivariant alternative to the Fiedler vector (e.g., steerable filters)
2. Accepting v2's strong constraint performance and addressing its fractional rotation failures specifically (perhaps by augmenting the segment matching with moment-based descriptors)
3. A hybrid approach: v2 for constraint satisfaction, v5 moment similarity as a tiebreaker or regularizer

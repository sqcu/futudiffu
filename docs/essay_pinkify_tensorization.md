# Tensorization of PINKIFY and THISNOTTHAT Reward Functions

## Date: 2026-02-18

## Summary

Added GPU-accelerated versions of `pinkify_score()` and `thisnotthat_score()` in
`src_ii/reward_functions.py`. Both functions now have three entry points:

- `pinkify_score_cpu(image)` / `thisnotthat_score_cpu(image, this, that)` -- original
  PIL/numpy/scipy implementation, unchanged.
- `pinkify_score_gpu(tensor)` / `thisnotthat_score_gpu(tensor, this_t, that_t)` -- pure
  torch tensor ops, supports [3, H, W] or [B, 3, H, W] input.
- `pinkify_score(image)` / `thisnotthat_score(image, this, that)` -- PIL-accepting
  wrapper that auto-dispatches to GPU when torch+CUDA are available.

## Speedup

RTX 4090 results:

| Function | Image Size | CPU (ms) | GPU (ms) | Speedup |
|----------|-----------|----------|----------|---------|
| pinkify | 256x256 (challenge) | 25-35 | 0.81-0.85 | 30-42x |
| pinkify | 1280x832 (V2 single) | 74-91 | 0.84-1.80 | 47-104x |
| pinkify | 1280x832 (B=8 batch) | -- | 1.48/image | -- |
| thisnotthat | 256x256 | 22-25 | 0.77-0.93 | 25-33x |
| thisnotthat | 1280x832 | 73-80 | 1.21-2.40 | 32-64x |

The GPU path is consistently >10x faster than CPU. For 256x256 images, individual
scoring is under 1ms. For 1280x832 images, individual scoring ranges 0.84-1.80ms
with some variance from kernel scheduling. Batch scoring at full resolution
(1280x832, B=8) achieves 1.48ms/image.

## Numerical Agreement

### PINKIFY

Near-exact agreement. Max relative difference: 3.07e-5 across all test images.
The only source of numerical divergence is float32 accumulation ordering differences
between numpy (sequential) and torch CUDA (parallel reductions). At the 8-digit
level shown in validation output, most scores match to all displayed digits.

Challenge set scores match to within 2e-7 relative tolerance. All V2 images match
within 3.1e-5. Spearman rank correlation: 1.000 (perfect).

### THISNOTTHAT

Larger but acceptable divergence. Max relative difference: 5.5%. The dominant cause
is the resizing method difference: the CPU path uses PIL `Image.LANCZOS` (windowed
sinc, high quality) while the GPU path uses `F.interpolate(mode='bilinear')`.
Lanczos is not available in `F.interpolate`, and implementing a custom Lanczos
kernel is not worth the complexity for a reward function.

Despite the numerical difference, **ranking is perfectly preserved**: Spearman
rho = 1.000 across all test images. The resizing difference shifts all scores
uniformly and does not change their relative ordering.

## Implementation Details

### RGB to HSV (torch)

Standard algorithm implemented with `torch.where` for the hue conditionals. The key
subtlety: the numpy implementation uses sequential masked assignment (`h[mask_r] = ...;
h[mask_g] = ...`) where later writes overwrite earlier ones at overlap pixels. The
torch version replicates this by applying masks in the same order (r, g, b) without
mutual exclusion, so `torch.where(mask_b, h_b, ...)` is applied last and overwrites.

### Coverage Contrast (torch)

Replaces `scipy.ndimage.uniform_filter(mode='reflect')` with:

```python
padded = F.pad(presence, (pad, pad, pad, pad), mode='reflect')
local_fraction = F.avg_pool2d(padded, kernel_size=7, stride=1, padding=0)
```

`F.avg_pool2d` with `stride=1` computes a box filter, equivalent to the uniform
filter. Reflect padding is applied manually before the pool.

### Cosine Similarity (torch)

Uses `F.cosine_similarity(a_flat, b_flat, dim=1)` on flattened (B, 3*H*W) tensors.
Handles batch broadcasting: reference images are (1, 3, H, W), expanded to match
the batch dimension.

### Reference Image Caching

`thisnotthat_score_gpu` accepts pre-computed reference tensors. The PIL wrapper
`thisnotthat_score()` uses a module-level cache keyed by `(device, H, W)` to avoid
re-resizing references on every call.

## Ranking Validation

### PINKIFY Challenge Set (GPU scores)

| Image | GPU Score | Required |
|-------|-----------|----------|
| A | 0.00000000 | Lowest |
| B | 0.00706966 | > A |
| C | 0.00759368 | > B |
| D | 0.00898744 | > C |
| E | 0.00898744 | = D |
| F | 0.04615732 | > D, E |

All 7 ranking checks pass: A < B < C < D, D = E, {A,B,C} < {D,E} < F.

### V2 Images

Spearman rho = 1.0 across 10 V2 images for both pinkify and thisnotthat.

## Batch Support

Both GPU functions accept `[B, 3, H, W]` input and return `[B]` scores. Verified
that batch scoring produces identical results to individual scoring (max difference
< 1e-5).

## No New Dependencies

The GPU path uses only `torch` and `torch.nn.functional`. No scipy, no torchmetrics,
no external packages. The `import torch` is wrapped in a try/except so the file
still works in environments without torch (falls back to CPU path).

## Files Modified

- `src_ii/reward_functions.py`: Added GPU implementations and auto-dispatch wrappers.
  Original CPU functions renamed to `pinkify_score_cpu` / `thisnotthat_score_cpu`.
  New `pinkify_score_gpu` / `thisnotthat_score_gpu` for direct tensor access.
  Existing `pinkify_score` / `thisnotthat_score` now auto-dispatch.

## Validation Artifacts

- Script: `scripts_ii/validate_gpu_reward_functions.py`
- Results: `pinkify_test_output/gpu_validation/gpu_validation_results.json`

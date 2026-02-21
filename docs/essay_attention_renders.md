# What the Attention Maps Show

*Generated 2026-02-16. Renders in `pinkify_thisnotthat_output/attention_maps/renders/`.*

## What was captured

The interpretability run (`scripts_ii/attention_interpretability.py`) selected four latents from the BTRM dataset and ran each through the Z-Image NextDiT backbone while monkey-patching `sdpa_attention` to record per-token attention statistics at every layer. The four latents represent extremes of the two reward heads:

- `high_pinkify`: traj 5, step_09. Peak "pinkify" BTRM score (0.0925).
- `low_pinkify`: traj 3, final step. Minimum pinkify score (0.0001).
- `high_thisnotthat`: traj 2, final step. Maximum "thisnotthat" score (0.1266).
- `low_thisnotthat`: traj 7, final step. Minimum thisnotthat score (-0.0175).

For each layer the capture reduced attention weights to:
- `attn_received[head, token]`: mean attention weight that each token receives (averaged over all query positions). This answers "which tokens are being looked at?"
- `attn_given[head, token]`: mean attention that each token gives (averaged over keys). This answers "which tokens are doing the looking?"
- `head_norms[head]`: L2 norm of the full attention matrix per head.

The diff files (`attention_diff_pinkify_diff.pt`, `attention_diff_thisnotthat_diff.pt`) contain element-wise differences between the high- and low-scoring samples for the same reward head. These isolate which tokens changed attention behavior in the direction correlated with reward.

## What the rendered images show

### `decoded_*.png`

VAE-decoded versions of the captured latents. These are the actual noisy intermediate images the attention statistics were taken from — not final denoised images, except for the `final` step samples which are fully denoised. The `step_09` latent (high_pinkify) is noticeably noisier; the `final` step latents look like completed images.

### `overlay_*.png` (hot colormap)

Per-spatial-token attention received, aggregated across all 30 main DiT layers and all 30 heads, overlaid on the decoded image at alpha=0.4. The hot colormap (black → red → yellow → white) encodes raw attention magnitude.

What this shows: the model's aggregate attention focus on different spatial regions of the image. Bright (hot) regions received more attention from other tokens on average. Attention tends to concentrate on high-frequency edges, texture boundaries, and semantically salient regions rather than flat low-entropy areas. The patterns are largely consistent across the high/low pinkify pair — the base model (no reward adapter) attends similarly to the same spatial structures regardless of whether the image's reward score is high or low. Differences are subtle.

### `overlay_pinkify_diff.png` and `overlay_thisnotthat_diff.png` (diverging colormap)

Attention-received difference heatmaps: high-scoring sample minus low-scoring sample, aggregated over 30 layers and 30 heads, then overlaid on the high-scoring image with a blue-white-red diverging colormap. Red regions received more attention in the high-reward sample; blue regions received more attention in the low-reward sample.

The pinkify diff shows modest but spatially structured differences. The magnitude is small (`mean_abs_received_diff=0.000176`), consistent with the expectation that both samples use the same base model weights with no reward adapter injected — the attention difference here is purely due to the different content of the two latents, not any trained reward signal. The max single-token received diff is 0.955 (essentially a sparse spike at one token), but the mean is in the fourth decimal place.

The thisnotthat diff is slightly larger (`mean_abs_received_diff=0.000204`), with top-diff heads at indices 19, 8, 22, 9, 3 (pinkify: 8, 16, 10, 7, 9). Different head sets respond most to the two reward axes, which is interesting: the thisnotthat and pinkify conditions activate partly different sets of heads as the most-changed by the content difference.

### `layer_head_*.png`

A 30-layer × 30-head heatmap of mean absolute attention-received diff (`|received_diff|.mean(dim=seq)`). Each cell shows how much a given head in a given layer differed between the high- and low-reward samples. The intensity (red = large diff, white = small diff) reveals which layers and heads are most sensitive to the reward-correlated content variation.

Both diff conditions show the largest differences concentrated in the middle-to-late main layers (layers 10-25 of the 30 main blocks, corresponding to global indices 14-29). Early layers show smaller diffs. This is consistent with the known role of later transformer layers in capturing higher-level semantic content, which is what pinkify and thisnotthat probe.

### `text_attn_*.png`

Bar charts of per-text-token attention received, averaged over all 30 main layers and all 30 heads. The sequence layout is: [caption tokens (first)] + [image tokens (remaining)]. With seq_len=4192 and 4160 image tokens (52×80 grid, padded to multiple of 32), there are 32 caption tokens. With seq_len=4224 (low_thisnotthat), there are 64 caption tokens — a longer prompt.

The charts confirm that text tokens receive non-trivial attention even when averaged in. The bar patterns are roughly uniform across positions, with minor variation per token, suggesting the model is not strongly localizing on specific text tokens when processing these particular latents.

### `text_diff_*.png`

Per-text-token attention-received difference (high minus low). Small but nonzero, indicating that even with the same model weights, different image content causes marginally different attention routing back to the caption tokens. No single caption token dominates.

### `composite_*.png`

Side-by-side pairing of the decoded image (left) and its attention overlay (right) at full resolution (2560×832). Useful for directly comparing spatial attention structure with image content.

### `summary_all_overlays.png` and `summary_diff_overlays.png`

Thumbnail strips of all four per-image overlays and both diff overlays. Quick overview of all conditions.

## What the maps do not show

These maps capture attention statistics from the base model with no reward adapter loaded. They characterize the attention geometry of the unconditional backbone on different latent inputs. They do not show:

- How a trained rtheta LoRA adapter *changes* attention routing. To see that, you would re-run the capture with the rtheta adapter active and compute a second round of diffs (adapter-on minus adapter-off for the same latent).
- Per-denoising-step evolution of attention. The captures are single-step snapshots at arbitrary timesteps in the trajectory (step_09 for high_pinkify, final step for the rest).
- Causal claims about what spatial regions *cause* high reward. The attention structure is correlated with image content and reward scores but the relationship is observational.

## Technical note: layer indexing

The NextDiT architecture has 34 total attention-calling blocks: 2 context refiners (text-only, seq_len=32), 2 noise refiners (seq_len=4160), and 30 main blocks (seq_len=4192). The render script filters to only the 30 main blocks when aggregating `attn_received` for the overlay images. The diff files were produced with this same filtering applied during the interpretability run and contain only the 30 main layers already.

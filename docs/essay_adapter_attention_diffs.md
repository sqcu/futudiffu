# What the r_theta Adapter Does to Attention Routing

*Generated: 2026-02-16 23:04:48*

## Setup

This analysis compares two forward passes through the same frozen NextDiT backbone:
- **Forward A**: r_theta LoRA adapter scale = 0.0 (unadapted reference)
- **Forward B**: r_theta LoRA adapter scale = 1.0 (reward adapter active)

The r_theta adapter was trained by `BTRMCompoundModel` on the `pinkify` and
`thisnotthat` reward heads over the 10-trajectory pinkify dataset. It has rank-8
LoRA weights injected into all attention QKV+out and FFN w1/w2/w3 projections
across 30 transformer blocks.

The same 4 representative latents used in the base-model attention study
(`attention_interpretability.py`) were reused: high/low pinkify and
high/low thisnotthat scores at their extreme steps.

## Magnitude of Attention Change

The mean absolute difference in per-token attention received (averaged over
all heads and all sequence positions, across all main DiT layers) is:

- **high_pinkify**: mean |Δ| = 0.000008
- **low_pinkify**: mean |Δ| = 0.000007
- **high_thisnotthat**: mean |Δ| = 0.000009
- **low_thisnotthat**: mean |Δ| = 0.000008

Overall mean: **0.000008**  
Overall max: **0.0264**

For context: the base-model cross-latent attention differences (high vs low
pinkify, different input images) in the prior study had mean |Δ| ~ 0.000176
and max |Δ| ~ 0.955. The adapter-induced delta (same image, adapter on vs off)
is thus a self-contained measurement of how much the learned weights perturb
the routing on identical inputs.

## Which Heads Are Most Affected

The top attention heads (by mean absolute received-attention change) per latent:

- **high_pinkify**: heads [9, 7, 16, 17, 13]
- **low_pinkify**: heads [9, 16, 7, 8, 3]
- **high_thisnotthat**: heads [9, 16, 8, 7, 25]
- **low_thisnotthat**: heads [9, 16, 7, 8, 24]

These head indices index into the 30-head NextDiT attention, where each head
has dimension 128 (total dim 3840). The heads most perturbed by the adapter
are not uniformly distributed, suggesting the adapter has learned to route
through specific attention circuits rather than diffusely perturbing all heads.

## Spatial Structure of the Perturbation

The attention diff heatmaps (see `adapter_attention_diffs/overlay_adapter_diff_*.png`)
show the spatial distribution of the received-attention change across image tokens.
Blue regions receive *less* attention when the adapter is active; red regions
receive *more*.

The diverging colormap is normalized per-image, so the absolute scale varies.
The key question is whether the spatial pattern correlates with the reward signal:
- For the pinkify pair: do the high-pinkify and low-pinkify latents show
  different spatial patterns?
- For thisnotthat: does the adapter shift attention differently depending on
  whether the image is in the 'this' or 'that' category?

## Layer Depth Profile

The layer x head heatmaps (`layer_head_adapter_*.png`) show whether the adapter
affects early layers (0-10), middle layers (10-20), or late layers (20-30) most.

The LoRA rank-8 adapter has the same rank throughout all layers, but the effective
influence depends on how strongly each layer's attention is modulated by the
adapter's learned A and B matrices. If the effect concentrates in late layers,
the adapter is functioning as a final-stage routing corrector. If it concentrates
in early/middle layers, it is modulating the base representation before the
reward-relevant features are built.

## Interpretation Caveats

1. **Small training run**: The r_theta adapter was trained for only ~30 BTRM
   macrobatches + 50 policy iterations on 10 trajectories. The weights are
   unlikely to have converged to a mechanistically clean solution.

2. **init_b_std=0.01**: Small but nonzero B initialization means the adapter
   starts with a small random perturbation. The early-training signal may be
   dominated by initialization noise rather than learned structure.

3. **Defect 24 context**: The live run had a defect where rtheta LoRA was
   never trained (all lora_B = 0). BTRMCompoundModel was written to prevent
   this. These results use the compound model's fixed loading path, so the
   adapter weights here *were* trained.

4. **Attention != output**: The adapter modifies linear projections, not
   attention weights directly. The attention change seen here is an *indirect*
   effect: the adapter changes QKV outputs, which changes attention logits.
   A larger direct effect is on the FFN outputs (not measured here).

## Files

```
pinkify_thisnotthat_output/adapter_attention_diffs/
  adapter_diff_{name}.pt          -- per-layer diff tensors (bfloat16)
  stats_scale0_{name}.pt          -- raw attention stats, adapter off
  stats_scale1_{name}.pt          -- raw attention stats, adapter on
  decoded_{name}.png              -- VAE-decoded image
  overlay_adapter_diff_{name}.png -- spatial attention diff overlay
  text_diff_adapter_{name}.png    -- text token diff bar chart
  layer_head_adapter_{name}.png   -- layer x head heatmap
  composite_{name}.png            -- decoded + overlay side by side
  summary_adapter_diffs.png       -- all overlays in a strip
  run_manifest.json               -- full run metadata + summary stats
```

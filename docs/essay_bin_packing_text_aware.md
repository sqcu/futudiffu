# Text-Aware Bin Packing: How Ignoring Caption Tokens Caused Silent Over-Packing

**Date**: 2026-02-17
**Scope**: `src_ii/bin_packer.py`, FlexAttention sequence packing for mixed-resolution generation

---

## The Architecture Assumption That Was Wrong

The bin packer schedules how many images can share a single FlexAttention
forward pass. Its job is to pack multiple small images into the sequence
capacity of one reference-resolution (1280x832) kernel call, because
small images underutilize SM89 tensor cores and can ride for free
inside the same attention computation.

The original bin packer computed capacity using image patch tokens only:

```python
REFERENCE_SEQ_LEN = compute_seq_len(1280, 832)  # = 4160
```

A 256x256 image produces 256 patch tokens. 4160 / 256 = 16. So the packer
claimed 16 small images fit per bin.

This was wrong. The error is not in the arithmetic but in the model of
what occupies the sequence.

## NextDiT Is Not Cross-Attention

In architectures like Stable Diffusion's UNet, text conditioning enters
through cross-attention: the text tokens form a separate KV sequence that
does not consume capacity in the image self-attention layers. In that
world, text tokens are free -- they do not compete with image tokens for
sequence slots.

NextDiT (Z-Image's diffusion transformer) does not work this way. Text
tokens are **concatenated** with image patch tokens into a single
self-attention sequence:

```python
# From the actual forward pass (build_packed_sequence):
padded_full_embed = torch.cat([cap_feats_embedded, x_patches], dim=1)
```

Both text and image tokens share the same attention matrix. Both consume
sequence capacity. When the bin packer ignores text tokens, it
underestimates the true cost of each item in the packed batch.

With 45 caption tokens (padded to 64) and 256 image tokens, the true
per-item cost is 64 + 256 = 320, not 256. Sixteen items at 320 tokens
each consume 5120 tokens -- **960 tokens beyond the reference capacity
of 4160**. The packer promised they would fit; they do not.

## Why It Matters

Over-packing has two failure modes:

1. **OOM**: The FlexAttention kernel allocates memory for the actual
   sequence length. If the packed sequence exceeds what the reference
   kernel was compiled for, it triggers recompilation for the larger
   size (each new total_len costs 45-73 seconds of compile time on SM89)
   or, worse, exceeds VRAM and crashes.

2. **Block mask corruption**: FlexAttention's document_id masks are
   constructed based on the declared sequence layout. If the actual
   tokens exceed the declared capacity, the mask is wrong. Tokens from
   image N may attend to tokens from image N+1, producing silently
   corrupted output.

Neither failure mode produces an obvious error message. The packer
silently over-commits, and the failure manifests downstream as either
a CUDA OOM with no clear origin or subtly wrong generated images.

## The Measurement

Before choosing a fix, we needed to know the actual distribution of
caption token lengths. The prompts are not uniform -- they range from
terse descriptions ("An astronaut riding a horse...") to adversarial
inputs ("ahem. *ting ting ting ting ting*...") designed to stress the
text encoder.

`scripts_ii/measure_prompt_tokens.py` tokenized all 33 unique prompts
from the V2 BTRM dataset using the Qwen3-4B tokenizer with the Z-Image
chat template (`<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`).

Results from `bin_packing_audit/prompt_token_stats.json`:

| Statistic | Raw tokens | Padded to 32 |
|-----------|-----------|--------------|
| min       | 22        | 32           |
| max       | 113       | 128          |
| mean      | 34.7      | 64           |
| median    | 31        | 32           |
| p90       | 45        | 64           |
| p95       | 45        | 64           |
| p99       | 113       | 128          |

The distribution is bimodal. Most prompts are 22-35 tokens (simple
descriptions). A cluster of style-transfer prompts ("anime character
with long rabbit ears... as a stained glass window with bold lead
lines") land at 38-46 tokens. One adversarial prompt hits 113 tokens.

The key number: **p90 = 45 raw tokens, which pads to 64**. This means
90% of real prompts add exactly 64 tokens of overhead per item in a
packed batch.

## The Fix

The bin packer now computes the **effective** sequence length per item,
accounting for both text and image tokens with their respective padding:

```python
def compute_effective_seq_len(
    width, height, cap_tokens,
    pad_multiple=32, vae_scale=8, patch_size=2,
):
    img_raw = compute_seq_len(width, height, vae_scale, patch_size)
    img_padded = _pad_to_multiple(img_raw, pad_multiple)
    cap_padded = _pad_to_multiple(cap_tokens, pad_multiple)
    return cap_padded + img_padded
```

The padding is not optional. The packed sequence layout
(`build_packed_sequence`) pads each segment -- text and image
independently -- to multiples of 32 for alignment. The old code
implicitly assumed the padding only applied to image tokens; the new
code mirrors the actual tensor layout.

Three constants were updated:

```python
DEFAULT_CAP_TOKENS = 45   # p90 of measured real dataset

# Old: REFERENCE_SEQ_LEN = 4160 (image-only)
# New: includes text overhead
REFERENCE_TOTAL_LEN = compute_effective_seq_len(1280, 832, cap_tokens=45)
# = pad32(45) + pad32(4160) = 64 + 4160 = 4224
```

`REFERENCE_SEQ_LEN` (4160) is preserved for backward compatibility but
is no longer the default bin capacity. `BinPackScheduler.__init__`
defaults to `REFERENCE_TOTAL_LEN` (4224).

The `pack_generation_plan()` method now populates `effective_seq_len`
per item, using either the item's actual `cap_tokens` field (if the
caller has tokenized the prompt) or `DEFAULT_CAP_TOKENS` (45) as a
conservative estimate. The `build_generation_plan()` function accepts
an optional `cap_tokens_per_prompt` mapping to pass through measured
per-prompt token counts for precise packing.

## Impact on Packing Density

The validation audit (`bin_packing_audit/validation_report_v2.txt`)
compares old and new packing for every resolution tier:

| Resolution | Image-only items/bin | Text-aware items/bin | Delta |
|-----------|---------------------|---------------------|-------|
| 1280x832  | 1                   | 1                   | +0    |
| 832x1280  | 1                   | 1                   | +0    |
| 1024x1024 | 1                   | 1                   | +0    |
| 512x512   | 4                   | 3                   | -1    |
| 640x384   | 4                   | 4                   | +0    |
| 576x448   | 4                   | 3                   | -1    |
| 256x256   | 16                  | 13                  | -3    |
| 320x192   | 17                  | 13                  | -4    |
| 192x320   | 17                  | 13                  | -4    |
| 288x224   | 16                  | 13                  | -3    |

Full-resolution images are unaffected: a single 1280x832 image at 4224
effective tokens exactly fills the bin. The 64 tokens of headroom that
existed under the old scheme (4160 image + 0 text < 4160 capacity was
wrong; 4160 image + 64 text = 4224 capacity is exact) were always
being consumed, just unaccounted for.

Medium-resolution images lose at most one item per bin. Small images
lose 3-4 items per bin. This is the largest absolute change: 256x256
goes from 16 to 13 items, an apparent 19% density reduction.

## Why the Density Reduction Does Not Matter

The naive reaction to "13 instead of 16" is that throughput decreased
by 19%. This is wrong, for two reasons.

**First**, the old packing was invalid. Sixteen 256x256 images with
their text tokens would have consumed 5120 tokens in a 4160-capacity
bin. That is not 100% utilization; it is 123% utilization, which means
either a crash or silent corruption. The "throughput" of a corrupted
forward pass is zero.

**Second**, FlexAttention with block-diagonal masks means underfilled
bins are not wasted compute. Each packed image only self-attends to its
own tokens. The attention mask is block-diagonal by `document_id`, and
FlexAttention's block-sparse kernel **skips** masked-out blocks entirely
-- they are not computed and then zeroed, they are never computed at all.

The sparse compute ratio for a bin with items of effective lengths
s_1, s_2, ..., s_N and total capacity C is:

```
sparse_ratio = sum(s_i^2) / C^2
```

For 13 items of 320 tokens each in a 4224-capacity bin:

```
utilization  = 13 * 320 / 4224 = 98.5%
sparse_ratio = 13 * 320^2 / 4224^2 = 1,331,200 / 17,842,176 = 0.0746
```

That is, the actual attention FLOPS are only **7.5% of dense**. The
quadratic scaling of self-attention means that many small images packed
together are dramatically cheaper per-image than one large image at the
same total token count. The 1.5% of unused capacity in the bin
contributes essentially nothing to the cost.

Even at the worst case -- the medium tier at 77.3% utilization (3x
512x512 images, 3264 of 4224 tokens used) -- the sparse compute ratio
is only 0.199, meaning 80% of the dense attention cost is skipped.
The "wasted" 23% of capacity is not wasted compute; it is simply
unoccupied space in the sequence buffer.

The validation simulation over a realistic generation plan (140 items
from 10 prompts across all tiers and attention backends) shows:

- Old packing: 35 bins, 98.5% utilization
- New packing: 37 bins, 97.5% utilization

Two additional bins. In exchange, every bin is correctly sized and no
forward pass exceeds the reference sequence capacity.

## The Decision Criterion

The design target is **>=90% tenancy for p90 configurations**. "p90
configurations" means: using the 90th-percentile prompt length (45
tokens) with any resolution from any tier. "Tenancy" means the fraction
of bin capacity that is occupied by actual tokens.

At `REFERENCE_TOTAL_LEN = 4224`:

- Full-resolution (1280x832): 4224/4224 = **100%** tenancy.
- Small-resolution (256x256): 13 * 320 / 4224 = **98.5%** tenancy.
- Medium-resolution (512x512): 3 * 1088 / 4224 = **77.3%** tenancy.
- Medium-resolution (640x384): 4 * 1024 / 4224 = **97.0%** tenancy.

The 77.3% case (512x512 and aspect-ratio variants at 576x448, 448x576)
is below 90%. But the sparse compute ratio at 77.3% utilization is
0.199, meaning the actual FLOPS cost of that bin is only 20% of a
full-capacity dense computation. The unused capacity does not translate
to wasted compute because the block-sparse attention kernel skips it.

For the p99 case (113 tokens, padded to 128), the effective cost of
a 256x256 item rises to 128 + 256 = 384 tokens, yielding 11 items per
bin (384 * 11 = 4224, exactly full). This is a natural degradation for
outlier prompts and does not require special handling -- the packer
already accounts for per-item `cap_tokens` when provided.

## Summary of Changes

| Component | Before | After |
|-----------|--------|-------|
| Bin capacity constant | `REFERENCE_SEQ_LEN = 4160` | `REFERENCE_TOTAL_LEN = 4224` |
| Per-item cost | `compute_seq_len()` (image only) | `compute_effective_seq_len()` (text + image) |
| Text token budget | 0 (ignored) | `DEFAULT_CAP_TOKENS = 45` (p90 measured) |
| Text padding | Not modeled | `pad_to_32(cap_tokens)` per item |
| 256x256 items/bin | 16 (invalid) | 13 (correct) |
| 512x512 items/bin | 4 (invalid) | 3 (correct, some resolutions stay at 4) |
| Sparse compute visibility | Not computed | `estimate_sparse_compute_ratio()` and `_detailed()` |

The old `REFERENCE_SEQ_LEN` constant is retained for backward
compatibility (callers that explicitly pass it get image-only behavior).
New code should use `REFERENCE_TOTAL_LEN` and `compute_effective_seq_len()`.

## Broader Lesson

The defect was invisible because the bin packer and the FlexAttention
kernel were developed at different levels of abstraction. The kernel
knew about text tokens -- it concatenates `cap_feats_embedded` and
`x_patches` into one sequence. The packer did not -- it only counted
patch tokens. The two components agreed on the image token count but
disagreed on total sequence cost, and neither component had a check
that would surface the disagreement.

The fix is not just the corrected arithmetic. It is the introduction
of `compute_effective_seq_len()` as the single source of truth for
"how many tokens does this item actually cost in a packed batch,"
callable by any component that needs to reason about sequence capacity.
The measurement script (`scripts_ii/measure_prompt_tokens.py`) and the
validation audit (`bin_packing_audit/validation_report_v2.txt`) provide
the empirical grounding that the constant is not a guess but a measured
parameter of the real dataset.

# State of Affairs: Funfetti Batching for BTRM Training

**Date:** 2026-02-18
**Author:** Root session
**Provenance:** Synthesis of two Opus subagent audits of current BTRM training
pipeline, against user spec for multi-resolution packed training.

---

## 0. Why This Matters: Quantization-Aware RLAIF

The motivating goal of this repository is to demonstrate **RLAIF and policy
optimization strategies for quantization-aware training** by using reward
models instead of outcome distillation. The core thesis is that a reward
model trained on preferences between images generated under different
quantization regimes (FP8 GEMM, INT8 SageAttention QK, fused LoRA
kernels) can provide a gradient signal that teaches the policy adapter to
compensate for quantization artifacts — without ever needing a
full-precision teacher model.

This means the training path must use the **same kernels as inference**:
FP8 blockwise GEMM, fused SiLU+gate+FP8 requant, SageAttention with
INT8 QK quantization, multi-LoRA sparse routing. If the reward model
trains under SDPA but evaluates under SageAttention, it cannot learn to
discriminate the artifacts that SageAttention introduces. The training
kernels must be behaviorally identical to the inference kernels.

The current SDPA-only BTRM training runs (v2-v5) are **partial progress**:
they validate the training loop, loss computation, gradient flow, LR
scheduling, checkpoint infrastructure, and per-head accuracy tracking.
What they do not validate is kernel parity between training and inference.
The remaining work is integration — looping in the function design,
specification, and testing work already done for packed inference with
masked SageAttention.

The packed training path must support **both** backends:

1. **SDPA FlexAttention** — with mixed-precision custom kernels (FP8 GEMM,
   fused LoRA) and FlexAttention block masks for packed batches. This is
   the "reference" backend for correctness validation.

2. **SageAttention FlexAttention** — with masked Sage kernels
   (`_sage_attn_fwd_int8qk_bf16pv_masked`, full backward pass) and the
   same FP8/fused kernels. This is the "production" backend that matches
   inference behavior and enables the reward model to learn
   quantization-specific features.

Both backends must produce equivalent scores (within the established
divergence floor of ≤0.0625 max_abs/step) when given the same inputs.
The packed-vs-serial validation infrastructure already tests this for
inference; extending it to training is straightforward.

---

## 1. Where We Are

The BTRM differentiable training pipeline (`train_btrm_differentiable()` in
`src_ii/btrm_training.py`, orchestrated by
`scripts_ii/train_pinkify_differentiable.py`) processes **one image at a
time** through the full 6B backbone. Each optimizer step:

1. Samples one pair from ~1.6M combinatorial space
2. Loads two latents, each as `(1, 16, H/8, W/8)`
3. Runs two serial B=1 forward passes through 30 gradient-checkpointed
   transformer layers
4. Computes BT pairwise loss, backward, optimizer step

**1 pair → 2 NFEs → 1 optimizer step.** No batching. No packing. No
block masks. No SageAttention. The attention backend defaults to SDPA
because nobody calls `set_attention_backend()` in the training path.

The `src_ii/` modules for packed inference — `attention.py`,
`forward_packed.py`, `block_mask.py`, the masked SageAttention kernels in
`sage_kernels.py` — are **completely disconnected** from the training path.
They were built for inference generation, but they are designed to work
identically at training time. The masked SageAttention kernels have full
`register_autograd` backward passes. The FP8 GEMM and fused LoRA kernels
have autograd support (all 12 custom_ops). The gap is integration, not
capability.

The training path calls `extract_hidden_differentiable()` in
`btrm_model.py`, which calls backbone layers directly with
`block_mask=None`, bypassing all packed infrastructure.

### What already works

| Component | Status | Location |
|-----------|--------|----------|
| `BinPackScheduler` with FFD packing | Working | `src_ii/bin_packer.py` |
| `pack_for_training()` method | Written, **dead code** (never called) | `src_ii/bin_packer.py:501` |
| `forward_packed()` with mixed resolutions | Working for inference | `src/futudiffu/diffusion_model.py:1098` |
| Block mask construction (uint8 → FlexAttention) | Working | `src_ii/block_mask.py` |
| Masked SageAttention (int8qk + fp8qk variants) | Working, fwd + bwd | `src/futudiffu/sage_kernels.py` |
| Unified attention dispatch (sage/sage_masked/sdpa) | Working | `src_ii/attention.py` |
| Sigma shifting (SD3 Eq.23, per-image) | Working end-to-end | `src_ii/sigma_schedule.py` |
| Per-image RoPE cache construction | Working | `load_latent_fn` in training scripts |
| Gradient accumulation with loss scaling | Working | `btrm_training.py:699` |
| Backward crash guard (degenerate pairs) | Working | `btrm_training.py:700` |
| Packed vs serial validation test | Working | `scripts_ii/validate_packed_vs_serial.py` |

### What doesn't exist

| Missing piece | Why it matters |
|---------------|---------------|
| `score_differentiable_packed()` | No way to score N images in one packed forward with gradients |
| `extract_hidden_packed()` with gradient checkpointing | The load-bearing gap — must pack N images, run 30 grad-ckpt layers, unpack hidden states |
| Funfetti batch sampler (resolution-aware PDF) | Current pair sampler is resolution-agnostic but resolution-uniform — doesn't target the 33/67 megapixel/small split |
| Validation covariance across resolution/aspect/head indexings | Can't measure whether funfetti training is better without tracking accuracy per-resolution-bucket |

---

## 2. The Single-Stream Architecture Constraint

Z-Image NextDiT is a **single-stream diffusion transformer**. Every latent
is a fused sequence of:

1. **Text embeddings** — caption tokens, variable length per prompt,
   RMSNorm'd and projected to 3840-dim, refined through 2 context refiner
   blocks. Padded to multiple of 32.
2. **Image patch embeddings** — VAE latent `(C, H/8, W/8)` patchified to
   `(H/16 × W/16)` tokens of dim 3840, refined through 2 noise refiner
   blocks. Padded to multiple of 32.
3. **RoPE positional encoding** — 2D grid frequencies for image patches,
   1D frequencies for caption tokens, fused into a single
   `(1, 3, total_len, 1, rope_dim, 2)` cache. Resolution-dependent:
   different `(H, W)` → different RoPE grids.
4. **AdaLN modulation** — timestep embedding `(1, 256)`, shared across the
   full sequence. The same sigma modulates both text and image tokens
   within a single forward.

This means "batching" two images of different resolutions is not a simple
matter of padding tensors. The token counts differ per image, the RoPE
grids differ per image, the caption lengths differ per image. The only
correct way to batch them is FlexAttention with block-diagonal masks that
prevent cross-image attention.

This is exactly what `forward_packed()` does. It is the solved problem.
The unsolved problem is wiring it into the training path.

---

## 3. Why Imperfect Packing Is Fine

A recurring concern in batch packing is tenancy — what fraction of the
packed sequence is occupied by actual tokens vs padding. For 256×256
images (64 image patches after patchify), the bin packer achieves ~13/16
tenancy (81%) when packing into the reference sequence length of 4224
tokens.

**This concern is misplaced for FlexAttention.** Block masking is an a
priori known sparsity pattern. The Triton kernels skip masked blocks
entirely — there is no computation on cross-image attention blocks, and
padding blocks within the sequence only waste the linear (non-attention)
FLOPS. The quadratic attention FLOPS — which dominate for long sequences —
respect the block mask exactly.

Concretely: four 512×512 images packed together have ~1024 image tokens
total. The attention cost is 4 × (256² attention within each image) =
262K attention elements, NOT (1024² cross-image attention) = 1,048K. The
block mask gives us a 4× FLOPS reduction over naively attending across
all tokens, and the packing gives us 4× the cases per forward pass
compared to serial. The product is a 16× improvement in cases per
attention-FLOP.

For smaller images the arithmetic is even more favorable. Eight 256×256
images (64 tokens each, 512 total) cost 8 × (64² attention) = 32K
attention elements vs one 1280×832 image at 4160² = 17.3M. That is a
540× ratio.

**80-85% bin packing efficiency is not a problem. It is a feature.**
The alternative is 100% tenancy at one image per forward, which wastes
the GPU on underutilized kernels for every sub-megapixel image.

---

## 4. The McCandlish Gradient SNR Argument

McCandlish et al. (2018) established that training efficiency depends on
the ratio of gradient signal to gradient noise, which improves with larger
effective batch sizes. The critical batch size `B_crit` is the batch size
at which one step of SGD is worth half as much as one step with infinite
batch size, in terms of loss improvement per example processed.

For BTRM pairwise ranking, each pair contributes an independent gradient
signal. The current regime — 1 pair per optimizer step, 2 serial B=1
forwards — is maximally noisy. The gradient from a single pair is a
high-variance estimator of the true pairwise ranking gradient.

Funfetti batching improves this in two ways:

1. **More pairs per macrobatch.** If a microbatch packs 4 images and we
   accumulate 2 microbatches, we get 4 pairs per optimizer step instead
   of 1 (or even more if we compute all pairwise combinations within a
   packed batch). The gradient noise drops as 1/√N.

2. **Resolution diversity as implicit regularization.** A macrobatch
   containing 1280×832, 512×512, and 256×256 images samples from a
   broader distribution of the image quality space. The model must learn
   features that generalize across resolutions, which reduces overfitting
   to resolution-specific artifacts. This is the mechanism behind v4's
   Phase 3 instability — the model memorized resolution-specific score
   magnitudes. Mixed-resolution training would prevent this.

The target distribution from the user's spec:

> ~33% of NFE FLOPS on megapixel images, ~67% spread across smaller
> resolutions (256², 320², 384², 512², 704², 1024²)

This is not a uniform distribution over resolutions. It is a
FLOPS-weighted distribution that ensures the model sees substantial
megapixel training signal (where the quality discrimination is hardest
and most valuable) while also seeing many more cases at smaller
resolutions (where the FLOPS per case are cheap). The expected number
of cases per microbatch rises to 3-5 under this distribution.

---

## 5. What Needs to Be Built

Most of the code already exists. The integration work is **composition
and glue coding**, not invention. The recent progress on verified
end-to-end reward model training (v2→v5 arc) has resolved the most
painful tech debt from the old `src/`+`scripts/` implementation: gradient
flow verification, crash guards, cosine LR scheduling, checkpoint
infrastructure, on-the-fly GPU scoring. What remains is wiring the
proven packed-inference code into the proven training loop.

### Layer 1: Packed differentiable scoring (the load-bearing piece)

`BTRMCompoundModel` needs a method that:

1. Takes N `(latent, timestep, conditioning, num_tokens, rope_cache)`
   tuples — heterogeneous resolutions
2. Calls `prepare_packed_state()` to build the packing layout, block mask,
   and packed RoPE
3. Runs embedding + context/noise refiners (under `no_grad`, same as
   `extract_hidden_differentiable` Phase 1)
4. Detaches and creates fresh autograd graph (same as Phase 2)
5. Runs 30 main transformer layers with gradient checkpointing **and
   block masks** — this is the same `torch.utils.checkpoint.checkpoint(
   layer, embed, x_mask, freqs_cis, adaln_input, block_mask=block_mask)`
   pattern, but now `block_mask` is not None
6. Unpacks the hidden state sequence back to per-image tensors
7. Applies ScoreUnembedder to each image's hidden state independently
8. Returns `(N, n_heads)` score tensor with full gradient connectivity to
   adapter parameters

The method must work with **both attention backends**:

- **SDPA FlexAttention:** The reference/correctness backend. Uses
  `F.scaled_dot_product_attention` with FlexAttention block masks. All
  custom kernels (FP8 GEMM, fused LoRA, fused SiLU+gate+requant) are
  active — the only thing that changes between training backends is the
  attention dispatch.

- **Masked SageAttention:** The production backend that matches inference
  behavior. Uses `_sage_attn_fwd_int8qk_bf16pv_masked` with full backward
  pass via `register_autograd`. This is the backend that enables the
  reward model to learn quantization-specific features — the whole point
  of quantization-aware RLAIF. The masked Sage kernels accept the same
  block mask format as FlexAttention, so the switch is a backend flag,
  not a code path change.

The `src_ii/attention.py` unified dispatch already handles this switching
via `attention_backend` parameter. The packed forward just needs to pass
the block mask through.

The model's `forward_packed()` already does steps 2-5 for inference. The
new method is `forward_packed()` + gradient checkpointing + hidden state
extraction + ScoreUnembedder application. The `extract_hidden_differentiable`
method already does gradient checkpointing + hidden extraction for the
unpacked case. The task is combining these two existing implementations.

**Validation:** The packed vs serial validation script
(`scripts_ii/validate_packed_vs_serial.py`) already confirms that packed
and serial forwards agree to ≤0.0625 max_abs per step. The same
validation framework extends to training: pack N images, score them,
compare against N serial scores. The scores should match within the
established divergence floor.

### Layer 2: Funfetti batch construction

The training loop needs to:

1. Sample K pairs per microbatch (not 1)
2. Collect the 2K images into a resolution-tagged set
3. Call `pack_for_training()` to bin-pack them into FlexAttention batches
4. For each bin, call `score_differentiable_packed()` to get all scores
5. Compute pairwise BT loss across all K pairs (the scores are already
   paired by the sampler, not by position in the batch)
6. Accumulate gradients across microbatches as usual

The pair sampler already handles this — `sample_pair()` is called K times,
each call is independent. The `load_latent_fn` already builds per-image
RoPE caches. The `preference_fn` already scores on-the-fly. The only new
code is the bin-packing step and the multi-score forward.

### Layer 3: Resolution-aware sampling PDF

The pair sampler currently selects trajectories uniformly. For funfetti
batching, the sampling PDF should be parameterized to target the desired
FLOPS allocation:

- ~33% of NFE FLOPS on megapixel images (1024² and above)
- ~67% spread across 256² through 704²

This translates to a non-uniform trajectory selection weight proportional
to `desired_flops_fraction / actual_flops_per_image`. Small images get
oversampled (many cases per NFE-FLOP), large images get sampled at their
natural rate.

The PDF should be deterministic given a seed (for reproducibility) and
should degrade gracefully when the dataset lacks certain resolutions (fall
back to available resolutions, don't crash).

### Layer 4: Validation covariance infrastructure

Track per-step metrics indexed by:
- Resolution bucket (256², 384², 512², 704², 1024², 1280×832)
- Aspect ratio (scalar W/H)
- Head name (pinkify, thisnotthat)
- Sigma/logSNR bucket
- Trajectory source (original V1, GPU rollout, policy rollout)

Compute running covariance across all indexings using torch. The goal is
to answer: "does the trained BTRM have better validation accuracy on
1024×1024 reference cases per NFE under funfetti sampling vs monotonic
megapixel sampling?"

If funfetti-sampled training matches monotonic loss descent per iteration,
it is **totally dominant** in wallclock and FLOPS due to the
hardware-efficient sparsity advantage of FlexAttention block masking on
smaller images.

---

## 6. What Does Not Need to Be Built

Everything below is **implemented, tested, and has autograd support for
training backward passes** (all 12 custom_ops have `register_autograd`):

- **FlexAttention packed forward.** `forward_packed()` works. It handles
  mixed resolutions, per-image RoPE, block-diagonal masks.
- **Block mask construction.** `src_ii/block_mask.py` builds uint8 masks
  from packing info and converts to FlexAttention block masks.
- **Masked SageAttention with full backward.** Forward and backward
  kernels exist for both int8qk and fp8qk variants, with
  `HAS_BLOCK_MASK: tl.constexpr` that compiles away to zero overhead
  when False. The backward pass is registered via `register_autograd`
  and tested — these are training-ready kernels, not inference-only.
- **FP8 GEMM kernels with autograd.** Blockwise matmul, addmm,
  act_quant, fused SiLU+gate+FP8 requant — all have backward passes.
  These are the same kernels used at inference; using them at training
  time is the definition of quantization-aware training.
- **Multi-LoRA sparse routing with autograd.** Per-batch adapter
  dispatch kernel with backward pass. Enables different LoRA adapters
  per image in a packed batch (relevant for multi-adapter experiments).
- **Bin packer.** `BinPackScheduler` with FFD algorithm, text-aware
  sequence length computation, reference length of 4224 tokens.
  `pack_for_training()` is written and waiting to be called.
- **Sigma shifting.** `resolution_shift()` and `time_snr_shift()` are
  canonical imports used throughout.
- **Gradient accumulation.** Loss scaling by `1/grad_accum_steps` and
  the backward crash guard are both in place.
- **Packed vs serial validation.** 6-phase validation script confirms
  agreement within rounding error (≤0.0625 max_abs/step, cos_sim 0.987+).
  Extends naturally to training validation.

---

## 7. Implementation Feasibility

This is integration glue coding with a bit of debugging, not kernel
development or architectural research. The necessary code has been
written and validated: packed forward, block masks, masked attention
kernels with autograd, bin packing, sigma shifting, gradient
checkpointing, packed-vs-serial validation. The recent v2→v5 training
arc has resolved the hardest tech debt (gradient flow verification,
detach defect, crash guards, LR scheduling, checkpoint infrastructure).
What remains is composing these proven pieces.

The delegation pattern: a Claude Opus subagent steers Claude Sonnet 4.6
subsubagents. The Opus agent holds the reading list and rubric (this
document + the lifecycle spec + the existing module APIs). The Sonnet
agents do the file-level work:

1. **Sonnet A:** Write `score_differentiable_packed()` in
   `btrm_model.py`, composing `prepare_packed_state()` +
   gradient-checkpointed packed forward + hidden state unpacking +
   ScoreUnembedder. Must support both SDPA and SageAttention backends
   via `src_ii/attention.py` dispatch. Validate against serial scoring
   on 2-4 test images of mixed resolutions.

2. **Sonnet B:** Wire `pack_for_training()` into the training loop in
   `btrm_training.py`. Modify the microbatch loop to sample K pairs,
   collect 2K images, bin-pack, and call the packed scoring method.
   The training loop must handle variable-size microbatches (different
   number of pairs depending on resolution mix).

3. **Sonnet C:** Add per-resolution accuracy tracking and covariance
   computation to the training metrics. Extend the analysis scripts to
   report accuracy-per-NFE and accuracy-per-wallclock across resolution
   buckets.

These are parallelizable — A must complete before B (B calls A's method),
but C is independent. The total integration surface is small: one new
method on BTRMCompoundModel, one modification to the training loop, one
extension to metrics. The heavy lifting (attention kernels, packing,
validation) is already done.

---

## 8. Dataset Generation Prerequisites

The current V2 dataset is 96% 1280×832 (249/259 trajectories). Funfetti
training needs resolution diversity. Before the training integration can
be exercised at scale, we need a generation run that produces trajectories
at 256², 320², 384², 512², 704², and 1024² (or a subset).

The `run03_btrm_training.py` script is already configured for
`["full", "medium"]` tier generation (1280×832 + 512×512). Extending to
additional tiers is a configuration change, not a code change — the bin
packer and sigma shifting infrastructure handle arbitrary resolutions.

The generation can run concurrently with the integration work. By the
time the packed training forward is validated, the multi-resolution
dataset should be ready.

---

## Appendix: Resolution/FLOPS/Packing Reference Table

| Resolution | Pixels | Image tokens | Attention FLOPS ratio vs 1280×832 | Max pack into 4224 tokens | Cases per megapixel-equivalent NFE |
|-----------|--------|--------------|----------------------------------|--------------------------|-----------------------------------|
| 256×256 | 65K | 64 | 1/4225 | ~13 (81% tenancy) | ~13× |
| 320×320 | 102K | 100 | 1/1731 | ~10 (24% tenancy†) | ~10× |
| 384×384 | 147K | 144 | 1/835 | ~7 (24% tenancy†) | ~7× |
| 512×512 | 262K | 256 | 1/264 | ~4 (24% tenancy†) | ~4× |
| 704×704 | 496K | 484 | 1/74 | ~2 (23% tenancy†) | ~2× |
| 1024×1024 | 1049K | 1024 | 1/16.5 | 1 (24% tenancy†) | ~1× |
| 1280×832 | 1064K | 4160 | 1.0 (reference) | 1 (98% tenancy) | 1× (reference) |

† Tenancy computed as `(N × img_tokens + N × ~64_cap_tokens) / 4224`.
Actual tenancy depends on caption length variance. The point is that
FlexAttention block masking makes tenancy irrelevant to attention FLOPS —
the sparse kernel skips padding blocks.

> **Key insight:** 13 cases of 256×256 in one packed forward costs roughly
> the same attention FLOPS as 1 case of 1280×832, but provides 13×
> the gradient signal for the same compute budget. Even accounting for
> the linear (non-attention) FLOPS that don't benefit from sparsity,
> the win is substantial for a model where attention is the dominant cost.

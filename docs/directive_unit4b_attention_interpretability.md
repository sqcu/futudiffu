# Directive: Unit 4b — Attention Mechanistic Interpretability for BTRM Reward Head

## What This Tests

Given a trained BTRM reward head (r_theta) from Unit 4, visualize what the reward adapter is actually doing at the attention level. The core question: when r_theta is active vs inactive, which image regions and text tokens see the largest change in attention patterns? This lets us:

1. Verify the reward adapter is doing *something* (non-zero attention diff)
2. Verify it's doing something *sensible* (pinkify should attend to pink regions; thisnotthat should attend to structural features matching THIS/THAT)
3. Compare reward models to each other
4. Detect degenerate adapters (all zeros, or attention changes only in unrelated regions)

## The Measurement

For a given noisy latent at a given sigma:

1. **Reference forward**: Set all LoRA scales to 0 (no adapter effect). Run forward. Capture post-softmax attention weights at every layer.
2. **Reward forward**: Set r_theta LoRA scale to 1. Run forward. Capture post-softmax attention weights at every layer.
3. **Diff**: Per layer, per head: attention_diff = attn_reward - attn_reference. Shape: (n_heads=30, seq_len, seq_len) per layer.
4. **Reduce**: Per-layer full attention matrices are too large to store (2GB each). Reduce per layer to:
   - Per-token "attention received" diff: mean over the query dimension (how much more/less attention does each token receive from all others?)
   - Per-token "attention given" diff: mean over the key dimension (how much more/less attention does each token give to all others?)
   - Per-head L2 norm of the diff (which heads changed the most?)
   - Result per layer: (n_heads, seq_len) for received, (n_heads, seq_len) for given, (n_heads,) for head norms
5. **Store**: These reduced stats as persistent tensors, plus aggregate across layers.

## The Spatial Mapping

Every token in the sequence has a known identity:
- **Image tokens**: index 0..N_img-1. Each image token is a flattened patch. With patch_size=2, each token covers a 2x2 region in latent space. After VAE decode (8x upscale), each token covers a 16x16 pixel region. The spatial layout is row-major: token i maps to (row=i // W_patches, col=i % W_patches) in patch coordinates, where W_patches = latent_w // patch_size.
- **Text tokens**: index N_img..N_img+N_text-1. Each text token maps to a span in the tokenized prompt.

This mapping lets us render attention diffs as spatial heatmaps overlaid on the decoded image.

## What Gets Built

### A: Attention Capture Hook

A hook that registers on each attention layer's output (pre-O projection, post-softmax). Must work with the standard SDPA path (NOT SageAttention — SageAttention doesn't expose weights). The hook captures the attention weight matrix, computes the per-token reductions immediately (to avoid storing the full seq_len x seq_len matrix), and discards the full matrix.

Implementation approach:
- Register a forward hook on the attention computation in each of the 30+4 layers (30 main + 2 noise refiner + 2 context refiner)
- The hook receives the attention weights as an intermediate, computes per-token stats, stores them in a dict keyed by layer index
- IMPORTANT: Must work with the existing NextDiT architecture. Read `src/futudiffu/diffusion_model.py` to find where attention weights are computed. Look for `sdpa_attention` or `F.scaled_dot_product_attention` calls. The hook may need to be placed around the QKV computation to capture pre-softmax scores manually (since F.scaled_dot_product_attention doesn't expose attention weights by default).
- Alternative: use `torch.nn.functional.scaled_dot_product_attention` with `enable_math=True` or use the explicit QK^T/sqrt(d) -> softmax -> V path for this diagnostic. Performance doesn't matter (this is a diagnostic, not production).

### B: Dual Forward Script

`scripts_ii/attention_interpretability.py`:

1. Load FP8 diffusion model with r_theta adapter pre-allocated (scale=0 initially)
2. Load the trained BTRM r_theta weights from `pinkify_thisnotthat_output/trained_head.safetensors`
   - Note: Unit 4 trained a BTRM *head* (RMSNorm + Linear), not an r_theta LoRA adapter. The head operates on hidden states. For this diagnostic, we need to check: does the BTRM training process also train an r_theta adapter, or only the head? Read the training script and BTRM architecture to determine this.
   - If there IS a trained r_theta adapter: use it. Set scale=0 for reference, scale=1 for reward.
   - If there is NO trained r_theta adapter (only the head was trained): this diagnostic tests something different — it would show what attention changes the *head training* would need from an adapter. In this case, the immediate diagnostic is: show the attention patterns of the base model on latents that score high vs low on pinkify/thisnotthat, WITHOUT any adapter. This is still valuable — it shows what the model already "knows" about pink.
3. Pick 2-3 representative latents: one from a trajectory that scores high on pinkify, one low; one that scores high on thisnotthat, one low.
4. For each latent: run two forwards with attention capture, compute diffs.
5. Save per-layer per-token reduced stats to disk.

### C: Visualization Script

`scripts_ii/render_attention_maps.py`:

1. Load the reduced attention stats from Part B
2. Load the VAE, decode the latents to pixel space
3. For each latent:
   - Create a heatmap of "attention received" diff, aggregated across heads (or top-K most-changed heads), mapped to spatial positions on the decoded image
   - Create a bar chart or heatmap of text token attention diffs
   - Overlay the spatial heatmap on the decoded image (alpha blend)
   - Save as PNG
4. Create a summary image showing all latents side by side with their attention heatmaps

### Output

All to `pinkify_thisnotthat_output/attention_maps/`:
- Per-latent per-layer reduced attention stats (tensors)
- Per-latent decoded image with attention heatmap overlay (PNG)
- Per-latent text token attention diff visualization (PNG)
- Summary comparison image (PNG)
- Stats JSON with which layers and heads showed the largest diffs

## Execution Notes

- Use SDPA backend, NOT SageAttention (need access to attention weights)
- Use `--no-compile` for the diagnostic (hooks + attention weight capture may interact poorly with torch.compile)
- The model forward WITHOUT compilation takes ~20s per step on RTX 4090 — we only need 2-3 latents x 2 forwards each = ~2 minutes of inference
- Windows Python: `/mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe`
- FP8 weights: `F:\dox\ai\comfyui\ComfyUI\models\diffusion_models\z_image_fp8_blockwise.safetensors`
- VAE: `F:\dox\ai\comfyui\ComfyUI\models\vae\zimage.safetensors`

## Reading List

- `src/futudiffu/diffusion_model.py` — NextDiT architecture, where attention is computed
- `src/futudiffu/attention.py` — sdpa_attention function, attention backend switching
- `src/futudiffu/btrm.py` — ScoreUnembedder, what r_theta means
- `src/futudiffu/lora.py` — LoRA adapter allocation and scale control
- `pinkify_thisnotthat_output/trained_head.safetensors` — the trained BTRM head
- `pinkify_thisnotthat_output/per_image_scores.json` — which images score high/low
- `docs/essay_unit4_pinkify_thisnotthat.md` — what was trained and how

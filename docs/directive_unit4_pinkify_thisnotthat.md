# Directive: Unit 4 — PINKIFY and THISNOTTHAT

## What This Tests

This unit tests the BTRM training pipeline end-to-end, in isolation from any RL/policy gradient process. The test is: can we define trivially-computable pairwise preference rules, generate preference labels from real trajectory data, train a BTRM head to predict those preferences from diffusion model hidden states, persist the trained head, reload it, verify bit-for-bit equivalence, and apply it to score new data?

If this works, the BTRM pipeline is proven. If it doesn't, we know before touching REINFORCE.

## The Two Reward Heads

### PINKIFY (Head 0: "pinkify")

A pairwise preference: given two images, the PINKER one wins.

**Scoring function** (pixel-space, not differentiable through VAE, doesn't need to be):
1. Convert decoded image to a color space where "pink" is definable (HSV or LAB)
2. Define "pink" as a region in color space — high saturation, hue in the pink/magenta range, with contrast against non-pink surroundings
3. For each pixel, compute a "local pinkness" that accounts for contrast with neighboring non-pink pixels (a locally pink region surrounded by non-pink has higher signal than a uniformly pink image)
4. Score = total volume of pink-by-contrast across the image, with some prenormalization for image size

**Pairwise preference**: image A beats image B if pinkify_score(A) > pinkify_score(B)

### THISNOTTHAT (Head 1: "thisnotthat")

A pairwise preference: given an image, is it more similar to THIS or to THAT?

**Reference images** (on disk):
- THIS: `i2i_off_policies/pizza-ratto.png`
- THAT: `i2i_off_policies/offhand_pleometric.png`

**Scoring function**:
1. Resize and rescale THIS and THAT to match the dimensions of the judged image
2. Compute similarity(judged, THIS) and similarity(judged, THAT) — L2, cosine, SSIM, or any reasonable metric in pixel space
3. Score = similarity(judged, THIS) - similarity(judged, THAT)
4. Higher score = more like THIS, lower = more like THAT

**Pairwise preference**: image A beats image B if thisnotthat_score(A) > thisnotthat_score(B)

**The two-field effect**: This creates two score adjustment fields:
- Images with no semblance of THAT get nudged towards THIS
- Images similar to THAT and not THIS get pushed away from THAT

## What Gets Built

### Part A: Reward Functions (src_ii/)

Two Python functions that operate on decoded pixel-space images (PIL Images or uint8 tensors):

```
pinkify_score(image) -> float
thisnotthat_score(image, this_ref, that_ref) -> float
```

These are pure functions. No GPU required. No model involved. They implement the literal rules above.

### Part B: Preference Label Generation (scripts_ii/)

A script that:
1. Loads stored trajectory data (from btrm_dataset/ or validation_output_ii/)
2. VAE-decodes each stored latent to pixel space
3. Applies both scoring functions to each decoded image
4. Generates pairwise preference labels: for each pair of trajectory steps, which wins on pinkify? which wins on thisnotthat?
5. Writes the labels to a persistent file (JSON or parquet) alongside the step indices and scores

This produces the training data for Part C.

### Part C: BTRM Head Training (scripts_ii/)

A script that:
1. Loads the diffusion model (for hidden state extraction, NOT for inference)
2. Loads the stored latent checkpoints (noisy latents at each step)
3. Runs each noisy latent through the diffusion model backbone to extract hidden states
4. Trains a BTRM head with two outputs: ("pinkify", "thisnotthat")
5. Uses Bradley-Terry pairwise ranking loss against the preference labels from Part B
6. Trains for enough iterations that loss decreases measurably
7. Persists the trained head to disk (safetensors or state_dict)

### Part D: Persist/Load/Verify (scripts_ii/)

A script that:
1. Loads the persisted BTRM head from Part C
2. Runs the same scoring inputs through it
3. Compares output tensors to the pre-persist scores
4. Writes the comparison to disk — this is the bit-for-bit verification

### Part E: Score Distribution Visualization (scripts_ii/)

A script that:
1. Takes a set of trajectory latents
2. Scores them with both the trained BTRM head AND the literal rule functions
3. Writes a comparison showing how well the BTRM head's rankings agree with the literal rules
4. Produces a persistent output file (JSON with per-image scores from both sources)

## What This Does NOT Do

- No RL. No policy gradients. No REINFORCE. This tests the BTRM pipeline in isolation.
- No CFG. The BTRM backbone forward uses single-conditioning hidden state extraction, not the CFG sampling path.
- No generation of new images. This uses existing stored trajectory data only.

## Execution Notes

- The BTRM head architecture already exists in `src/futudiffu/btrm.py` (ScoreUnembedder)
- Hidden state extraction uses the HiddenCapture hook pattern from the existing code
- The diffusion model backbone must be loaded for hidden state extraction — this needs the RTX 4090's full VRAM
- VAE decode for preference label generation is separate (load VAE, decode, free, then load diffusion model)
- All outputs to `pinkify_thisnotthat_output/` at repo root

## Reading List for Implementing Agents

- `docs/unit4_user_spec_verbatim.md` — the user's original spec
- `docs/essay_algorithmic_decomposition.md` — the five function boundaries (nfe is how the model forward works)
- `docs/essay_codebase_analysis.md` — section on BTRM head architecture
- `src/futudiffu/btrm.py` — ScoreUnembedder class (RMSNorm → Linear → soft_tanh_cap)
- `src/futudiffu/training_utils.py` — how hidden states are currently extracted (the HiddenCapture hook, forward_checkpointed returning last_hidden)
- `src_ii/forward.py` — the nfe() function that wraps NextDiT.forward()
- `src_ii/model_loading.py` — how to load the diffusion model standalone
- `i2i_off_policies/pizza-ratto.png` and `i2i_off_policies/offhand_pleometric.png` — the THIS and THAT reference images

## Success Criteria

All of these produce persistent output files, not assertions:
1. Preference labels exist on disk with scores from both literal rules
2. BTRM training loss curve saved to disk, showing decrease
3. Persisted head exists on disk as a loadable artifact
4. Bit-for-bit comparison between pre-persist and post-load scores exists on disk
5. Score distribution comparison (BTRM vs literal rules) exists on disk, showing agreement

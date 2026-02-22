Alright. Using the notation from the threads:

```
CFG : OPERAND → NFE_POSTCFG

CFG(OPERAND) :=
  GATHER(
    SCATTER(OPERAND, {MODIFY_i, NFE_CONFIG_i}))
```

Where SCATTER and GATHER are:

```
SCATTER(OPERAND, branches) :=
  { NFE(MODIFY_i(OPERAND), NFE_CONFIG_i)  |  i ∈ branches }

GATHER({NFE_i}, {scale_i}) → NFE_POSTCFG
  where shape(NFE_POSTCFG) = shape(NFE_i) ∀ i
```

The full sampling step is then:

```
STEP(OPERAND) := SAMPLE(CFG(OPERAND), SAMPLE_CONFIG)
```

Now the 6tuple instantiation. Six branches, six MODIFY functions, six scales:

```
branches = {0, 1, 2, 3, 4, 5}

MODIFY_0(x) = x + cond_text_shrimp_color_shape
MODIFY_1(x) = x + cond_text_typographic_features
MODIFY_2(x) = x + cond_base, F.interpolate(↓ res_1)
MODIFY_3(x) = x + cond_base, F.interpolate(↓ res_2)
MODIFY_4(x) = x + cond_banana_poem
MODIFY_5(x) = x  [unconditional]
```

The critical constraint from the thread:

```
∀ i : shape(NFE(MODIFY_i(OPERAND), NFE_CONFIG_i)) = shape(NFE_POSTCFG)
```

Which means MODIFY_2 and MODIFY_3 run their NFE at *lower* resolution, then the *outputs* get `F.interpolate`d back up before GATHER sees them. The repulsive residual property comes from their negative scales in GATHER:

```
GATHER({nfe_0, ..., nfe_5}, {s_0, ..., s_5}) :=
  Σ_i  s_i * nfe_i

  where s_0, s_1 > 0     [positive guidance]
        s_2, s_3 < 0     [repulsive residual]
        s_4 < 0           [negative cfg]
        s_5 = baseline    [anchor term]
```

The fork-join graph:

```
              /-- MODIFY_0 -- NFE -- s_0 --\
             /-- MODIFY_1 -- NFE -- s_1 --\ \
            /-- MODIFY_2 -- NFE↓ -- ↑ -- s_2 --\
OPERAND ==>--- MODIFY_3 -- NFE↓ -- ↑ -- s_3 ----- GATHER ==> NFE_POSTCFG ==> SAMPLE
            \-- MODIFY_4 -- NFE -- s_4 --/
             \-- MODIFY_5 -- NFE -- s_5 --/
```

The design invariant from the last few posts: **nothing in this definition mentions the number 6**. The branch set `{0..5}` is a parameter. SCATTER maps over it. GATHER reduces it. The scatter-gather function is correct for `|branches| = 2` and `|branches| = 11_000_000` by the same rewriting rule.

The thing that's "written right or not at all" is: `cfg` never pattern-matches on branch count. It receives `{(MODIFY_i, scale_i, NFE_CONFIG_i)}` and maps/reduces. That's the whole content.

---

## Plain language: what is a guidance query?

A guidance query is a question you ask the diffusion model: "what would you
denoise this noisy image into, if this text described it?" The model answers
with a denoised estimate. You ask K questions per step, then combine the
answers into one update to the image.

Standard CFG asks 2 questions: "what do you see with the prompt?" and "what
do you see with no prompt?" The difference tells you what the prompt adds.
You amplify that difference.

The 6-tuple asks 6 questions: "what do you see with the shrimp description?"
"...with the typography description?" "...at half resolution?" "...at quarter
resolution?" "...with the banana poem?" "...with the base prompt?" Each answer
contributes to the update with a signed scale: positive pushes the image
toward that answer, negative pushes away from it.

## Why base = the main prompt, not unconditional

Standard CFG convention: base is the unconditional (empty text). The guided
estimate starts from unconditional and adds scaled residuals from the conditional.

Our convention: base is the main prompt (the actual thing you're generating).
The guided estimate starts from the main prompt's denoised estimate and adds
scaled residuals from other guidance queries. The main prompt's contribution
is always 1.0 — the identity. All other branches are deviations from it.

Why the flip: with K > 2 guidance queries, having the base be the actual
generation target is more natural. The "unconditional" becomes just another
guidance query with negative scale, not the privileged anchor.

## What scatter does physically

At each Euler step, you have one noisy image: the evolving base trajectory.
Scatter makes K copies of it — one for each guidance query. Same-resolution
copies are clones. Different-resolution copies are bilinear downsamples.

The model sees K independent images in a single packed forward pass (via
FlexAttention block masks — no cross-image attention leakage). Each copy
gets denoised with its own conditioning.

## What gather does physically

Gather takes K denoised estimates and combines them:

```
guided = base_denoised + sum(scale_i * (denoised_i - base_denoised))
```

Each `(denoised_i - base_denoised)` is a residual: what branch i produced
that's different from the base. Positive scale amplifies that difference
(attractive). Negative scale subtracts it (repulsive).

For different-resolution branches, the denoised output is upsampled back
to base resolution before computing the residual.

## The noise field: one big image, smaller queries see the center

At initialization (step 0), we don't use scatter. Instead, we generate
one master noise tensor at the maximum latent resolution. Each branch gets
a center-crop of this noise at its own resolution.

This means a 256x256 branch sees the center of the same noise structure
that the 1024x1024 branch sees. The low-res branch is denoising the same
spatial content, just at lower fidelity. This makes the repulsive residual
meaningful — it captures what the model loses at low resolution, not
unrelated noise.

After step 0, scatter takes over: the current base latent (which has been
partially denoised) is downsampled for lower-resolution branches.

## Why negative-scale low-res queries work

When the model denoises at 256x256, it hallucinates broad structure but
loses fine detail. The denoised estimate is blurry. The residual between
this blurry estimate and the full-resolution base estimate captures
"what the model sees when it can't see fine detail."

With a negative scale, we push the trajectory AWAY from this blurry
denoising. The effect: the trajectory moves toward finer spatial structure.
It's a differentiable "don't be blurry" signal.

Two resolution tiers (e.g. 512x512 and 256x256) let you separate
"medium-frequency hallucinations" from "all-frequency hallucinations" and
repel from each independently with different strengths.

## Concrete 6-tuple mapping

A prompt: "pink shrimp with crisp typography in a banana field."

```
Branch 0: "pink shrimp, detailed color, clear shape"
           1024x1024, scale = +1.0 (base trajectory)
Branch 1: "clean typography, sharp letterforms, no blur"
           1024x1024, scale = +3.0 (attractive: emphasize text clarity)
Branch 2: "pink shrimp, detailed color, clear shape"
           512x512,   scale = -2.0 (repulsive: push away from mid-res blur)
Branch 3: "pink shrimp, detailed color, clear shape"
           256x256,   scale = -1.5 (repulsive: push away from low-res blur)
Branch 4: "banana, yellow, tropical poem"
           1024x1024, scale = -4.0 (repulsive: push away from banana dominance)
Branch 5: "pink shrimp, detailed color, clear shape"
           1024x1024, scale = +2.0 (attractive: reinforce base)
```

The net effect per Euler step: the trajectory is pulled toward sharp shrimp
with clear typography and pushed away from blurry denoising and banana-dominated
compositions. Six model evaluations per step, combined into one Euler update.

Module: `src_ii/triumphant_future_reduction_ops.py`
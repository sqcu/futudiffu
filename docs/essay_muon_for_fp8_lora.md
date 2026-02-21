# Muon Optimizer for FP8 LoRA: Research and Feasibility Analysis

## 1. What Muon Is and Why It Helps Linear Projections

### The Core Idea

Muon (MomentUm Orthogonalized by Newton-schulz) is an optimizer designed
specifically for 2D weight matrices in neural network hidden layers. It was
introduced by Keller Jordan in late 2024 and landed in `torch.optim.Muon` as
of PyTorch 2.9.

The algorithm:

1. Accumulate standard momentum: `buf = beta * buf + grad`
2. Orthogonalize the momentum buffer via Newton-Schulz iteration
3. Apply Nesterov: `update = grad + beta * orthogonalized_buf`
4. Scale by `sqrt(fan_out / fan_in)` (classic) or `0.2 * sqrt(max(dims))` (KiMuon)
5. Apply to weights: `W -= lr * update`

The Newton-Schulz iteration computes the polar factor of the momentum matrix.
Given a matrix G, it finds the nearest (semi-)orthogonal matrix -- effectively
computing U*V^T from the SVD G = U*S*V^T, but without doing actual SVD. The
quintic iteration uses 5 steps with optimized coefficients:

```
(3.4445, -4.7750, 2.0315)  # KellerJordan reference implementation
```

or the DiON/Prime Intellect quintic coefficients:

```
(4.0848, -6.8946, 2.9270),
(3.9505, -6.3029, 2.6377),
(3.7418, -5.5913, 2.3037),
(2.8769, -3.1427, 1.2046),
(2.8366, -3.0525, 1.2012),
```

Each iteration: `X' = a*X + (b*A + c*A^2) @ X` where `A = X @ X^T`.

### Why This Helps Linear Projections

The insight is spectral. When you optimize a matrix W with SGD or Adam, the
update magnitude is controlled per-element. But what matters for a linear
projection is how the update affects the *spectrum* -- the singular values.
Adam's per-element adaptive LR has no awareness of the matrix structure.

Muon's orthogonalization collapses all singular values of the update to ~1,
giving direction-only steps. This means:

- All spectral directions receive equal update magnitude
- Small singular values (which Adam neglects) get the same attention as large ones
- No accumulator division instability (no v_t / sqrt(v_t) ratio)
- The optimizer state is just one momentum buffer (bf16-safe), not two (m + v)

Memory: Muon needs 2 bytes/param (bf16 momentum). AdamW needs 8 bytes/param
(fp32 m + fp32 v). This is a 4x reduction in optimizer state.

### State of the Art

Muon has achieved speedups of 1.35x for GPT-2 training to equivalent val loss,
and has been tested at 1B+ scale. It is now in PyTorch core and NVIDIA NeMo.
We have `torch.optim.Muon` available in our torch 2.10.0 environment.


## 2. Whether Existing Muon Implementations Handle FP8 or LoRA

### FP8 + Muon

The logsnrcat project (our sibling repo) has `FP8Muon` in `optim_fp8.py` which
operates directly on `FP8Linear` modules using STE-captured gradients. That
implementation uses per-tensor FP8 quantization with bf16 master weights. Our
futudiffu FP8Linear uses a different scheme: 2D blockwise quantization (128x128
blocks with per-block float32 scales), and crucially, the FP8 weights are
**frozen buffers** not trainable parameters. LoRA adapters are the trainable
parameters, not the base weights.

There is also a paper "Effective Quantization of Muon Optimizer States"
(arXiv:2509.23106) which shows 8-bit blockwise quantization of Muon's momentum
buffer itself, achieving ~74% memory reduction while matching full-precision
training quality.

Neither of these addresses our exact scenario: frozen FP8 base weights with
trainable bf16 LoRA adapters.

### Muon + LoRA Specifically

Two recent papers directly address this:

**"LoRA meets Riemannion" (arXiv:2507.12142, Jul 2025):** This is the most
relevant work. The key insight: standard Muon applied independently to LoRA A
and B matrices breaks reparameterization invariance. The same low-rank update
DW = A*B^T can be factored infinitely many ways, and optimizing A and B
separately in Euclidean space (even with Muon) is not invariant to this choice.

Riemannion solves this by optimizing on the fixed-rank manifold directly. It
maintains the A*B^T factorization for efficiency but computes updates using
Riemannian geometry: project the orthogonalized momentum onto the tangent space
of the rank-r manifold, then retract back. The projection requires QR
decompositions and an SVD of a 2r x 2r matrix (tiny when r = 8-64).

Results: +1-2% accuracy on commonsense reasoning benchmarks (Llama 3-8B, rank
16), with notable improvements on diffusion model fine-tuning (Stable Diffusion
2, ranks 4/8/16). Faster convergence than vanilla LoRA + Adam.

**"Uniform Spectral Growth of Muon in LoRA-Style Matrix Factorization"
(arXiv:2602.06385, Feb 2026):** This paper proves that even naive Muon on
separate A and B factors produces near-uniform growth of singular values in the
product A*B^T, despite orthogonalization being applied to each factor
independently. Smaller singular values actually reach their targets *earlier*
than larger ones -- the opposite of Adam's behavior. Global convergence is
guaranteed under l2 regularization.

This is encouraging: even the simple approach (Muon on A and B separately)
has better spectral properties than Adam.


## 3. What logsnrcat `optim_fp8.py` Does and Whether It's Adaptable

### Architecture

The logsnrcat `FP8Muon` (lines 577-722) is designed for a fundamentally
different scenario: **the FP8 weights themselves are trainable**.

The flow:
1. `FP8Linear` stores weights in FP8 with per-tensor scaling
2. Forward: dequantize via STE, compute in bf16, STE captures gradient
3. `FP8Muon.step()`: reads gradients from `_grad_holder` on each module
4. Accumulates momentum (bf16), orthogonalizes, applies update
5. `m.apply_update()` writes to bf16 master weight, then requantizes to FP8

The Newton-Schulz implementation is identical to our standalone Muon
(`optim_muon.py` in logsnrcat). Both use the DiON quintic coefficients.

### Adaptability to Our Case

Our futudiffu case is structurally different:

| Aspect | logsnrcat FP8Muon | futudiffu LoRA |
|--------|-------------------|----------------|
| What's optimized | FP8 weight buffers (via master) | bf16 LoRA A & B Parameters |
| Gradient source | STE `_grad_holder` | Standard autograd `.grad` |
| Base weight | Mutable (updated each step) | Frozen FP8 buffer |
| Weight shape | (out, in) e.g. (10240, 3840) | A: (r, in), B: (out, r) |
| Update path | `apply_update()` -> requantize | Standard `param.add_()` |

The orthogonalization code is directly reusable. The optimizer scaffolding is
not -- we need standard PyTorch optimizer that reads `.grad` from Parameters,
not one that reads from `_grad_holder` lists on FP8Linear modules.

This means we should use `torch.optim.Muon` directly on LoRA parameters, or
write a thin wrapper. We do NOT need to port `FP8Muon`.


## 4. What a "KiMuon for FP8 LoRA" Would Look Like

### The Parameter Landscape

For our NextDiT model (dim=3840, n_heads=30, ffn_dim_multiplier=8/3):

**LoRA targets (per layer, 5 linear layers per block):**

| Target | Base Shape | LoRA A Shape | LoRA B Shape |
|--------|-----------|-------------|-------------|
| attention.qkv | (5760, 3840) | (r, 3840) | (5760, r) |
| attention.out | (3840, 3840) | (r, 3840) | (3840, r) |
| feed_forward.w1 | (10240, 3840) | (r, 3840) | (10240, r) |
| feed_forward.w2 | (3840, 10240) | (r, 10240) | (3840, r) |
| feed_forward.w3 | (10240, 3840) | (r, 3840) | (10240, r) |

Note: w1 and w3 are fused into w1w3 (20480, 3840) for FP8 inference, but
LoRA injection happens before fusion, so the LoRA shapes correspond to the
original linear definitions.

At rank r=8:
- A matrices: (8, 3840) to (8, 10240) -- **very wide** (aspect ratio 480:1 to 1280:1)
- B matrices: (3840, 8) to (10240, 8) -- **very tall** (same aspect ratios, transposed)

At rank r=64:
- A: (64, 3840) to (64, 10240) -- aspect ratio 60:1 to 160:1
- B: (3840, 64) to (10240, 64) -- same transposed

**rtheta (BTRM reward model):** Only last 2 layers, rank 8 = ~594K params.
**ptheta (policy model):** All 32+2+2=36 layers, rank 8 = ~10.1M params.

### Does Muon's Newton Step Make Sense for Low-Rank Matrices?

This is the key question. The Newton-Schulz iteration computes the polar
decomposition of the momentum matrix. For a matrix of shape (m, n) where m << n
(or m >> n), the iteration is:

```
X = momentum.bfloat16()
if X.size(-2) > X.size(-1):
    X = X.mT  # always work with the wide matrix
X = X / X.norm()  # spectral norm <= 1
for a, b, c in NS_CONSTS:
    A = X @ X.mT    # (min_dim, min_dim) -- CHEAP!
    B = b * A + c * (A @ A)
    X = a * X + B @ X
```

The critical observation: **the inner product `X @ X.mT` has shape
`(min(m,n), min(m,n))`**. For rank-8 LoRA:

- `A @ A.mT` is (8, 8) -- trivially cheap
- `B.mT @ B` is (8, 8) -- same

The matmul `B @ X` is (8, 8) @ (8, 3840) = (8, 3840). Five iterations of
this is essentially free compared to the forward pass through a 6B model.

**Cost per NS step for a rank-8 A matrix (8, 3840):**
- `X @ X.mT`: 8 * 8 * 3840 = 245K FLOPs
- `A @ A`: 8 * 8 * 8 = 512 FLOPs
- `B @ X`: 8 * 3840 * 8 = 245K FLOPs
- Total per step: ~0.5M FLOPs
- 5 steps: ~2.5M FLOPs

For comparison, one forward pass through the model is ~12T FLOPs. The NS
iteration overhead is <0.001%.

At rank 64, the inner dimension is still only 64, giving ~40M FLOPs per NS
iteration -- still negligible.

**However**, the spectral structure question is more subtle. An (8, 3840)
matrix has at most 8 nonzero singular values. Orthogonalization maps these to
~1. That means the update direction is fully described by just 8 spectral
modes. Is that useful?

Yes, and here's why: the *gradient* of the loss with respect to A has shape
(8, 3840) and also has at most 8 nonzero singular values. Adam would apply
per-element scaling to these 8*3840 = 30,720 entries independently, destroying
the spectral structure. Muon preserves it. The spectral growth paper proves
this leads to uniform growth of the product's singular values.

For B matrices: same argument, transposed. Shape (10240, 8) has 8 singular
values. NS iteration works on the (8, 8) Gram matrix.

### Memory Overhead vs AdamW

Per-parameter state storage:

| Optimizer | State per param | For r=8 ptheta (10.1M params) |
|-----------|----------------|-------------------------------|
| AdamW | 8 bytes (fp32 m + fp32 v) | 80.8 MB |
| Muon | 2 bytes (bf16 momentum) | 20.2 MB |
| Riemannion | ~6 bytes (tangent basis + momentum) | ~60.6 MB |

Muon saves **60.6 MB** over AdamW for ptheta alone. For rtheta (594K params),
the saving is smaller (4.75 MB -> 1.19 MB) but the BTRM head parameters would
still need AdamW (they're not 2D projections).

The NS iteration itself needs a temporary (min(m,n), max(m,n)) bf16 buffer for
X, plus a (min(m,n), min(m,n)) buffer for A. At rank 8, this is 8*10240*2 +
8*8*2 = 163 KB per layer. Trivial.

### What A Concrete Implementation Would Look Like

Option A: **Use `torch.optim.Muon` directly on LoRA parameters.**

```python
from torch.optim import Muon

ptheta_params = list(get_lora_params(model, adapter_name="ptheta"))
optimizer = Muon(
    [{'params': ptheta_params}],
    lr=0.02,
    momentum=0.95,
)
```

This is the simplest approach. `torch.optim.Muon` already handles 2D
parameters via Newton-Schulz and falls through to plain momentum for 1D.
All our LoRA parameters are 2D, so every parameter gets orthogonalized.

This applies Muon to A and B matrices independently, which the spectral
growth paper shows still yields uniform singular value growth in the product.

Option B: **Riemannion-style manifold optimization.**

This would operate on the combined rank-r update rather than separate factors.
More theoretically principled but requires:
- Custom optimizer maintaining tangent-space state
- QR decompositions per step (of shape (r, r) -- cheap)
- Retraction back to the manifold
- Cannot use `torch.optim.Muon` off-the-shelf

Riemannion showed +1-2% accuracy over vanilla LoRA + Adam on commonsense
benchmarks. Whether this translates to our BTRM/policy training is unknown.

Option C: **KiMuon variant for AdamW-compatible LR.**

Same as Option A but with `0.2 * sqrt(max(dims))` scaling plus decoupled
weight decay, allowing reuse of our existing LR schedules. This is what
logsnrcat's `Muon(variant="kimuon")` implements.


## 5. Recommendation

### Worth implementing: YES, and the path is nearly trivial.

**For immediate deployment (this week):**

Use `torch.optim.Muon` directly on LoRA parameters. This is a **two-line
change** in `trainer.py`:

```python
# Before:
policy_optimizer = torch.optim.AdamW(ptheta_params, lr=config.policy_lr)

# After:
policy_optimizer = torch.optim.Muon(
    [{'params': ptheta_params}],
    lr=0.02,  # Muon needs ~100x higher LR than AdamW
    momentum=0.95,
)
```

And similarly for btrm_optimizer (LoRA params only; the ScoreUnembedder
parameters are 1D/small-2D and should stay on AdamW).

**Expected benefits:**

1. **Memory**: 4x less optimizer state (bf16 momentum only vs fp32 m+v).
   For ptheta: 80.8 MB -> 20.2 MB. Not life-changing on a 24GB card running
   a 6B model, but it helps at the margins.

2. **Training speed**: The spectral growth paper proves Muon gives uniform
   singular value growth in LoRA factorizations. Adam preferentially amplifies
   the largest singular values, which means small but important spectral modes
   are neglected. This is especially relevant for rank 8 where you only have 8
   modes total -- you cannot afford to waste any of them.

3. **LR sensitivity**: Muon is known to be less sensitive to LR choice than
   Adam. This matters when we're doing 50-iteration policy optimization where
   we don't have budget for hyperparameter search.

4. **Numerical stability**: No v_t accumulator means no division by near-zero.
   bf16 momentum is safe (unlike Adam where bf16 v causes instability).

**Expected costs:**

1. Newton-Schulz overhead: <0.001% of forward pass FLOPS. Negligible.
2. LR tuning: Muon's LR is ~100x AdamW's. Need to find the right value.
   KiMuon variant avoids this by normalizing update RMS.
3. Integration testing: Need to verify that gradient clipping + Muon
   interaction is correct (clip the gradient before Muon accumulates it).

**For future work (if results are promising):**

Investigate Riemannion for manifold-aware LoRA optimization. The paper shows
consistent improvements on diffusion model fine-tuning. The implementation
complexity is moderate (QR + small SVD per step, both on (2r, 2r) matrices)
but requires a custom optimizer class rather than using torch.optim.Muon.

### What NOT to do:

1. Do NOT port `FP8Muon` from logsnrcat. It's designed for mutable FP8 weights,
   not frozen-base + trainable-LoRA. Wrong abstraction.

2. Do NOT apply Muon to the ScoreUnembedder. It has an RMSNorm (1D) and a
   Linear(3840, 2) which is technically 2D but has only 2 output features.
   Orthogonalizing a (2, 3840) matrix is well-defined but questionable -- the
   update lives in a 2D subspace and orthogonalization just normalizes the two
   rows. AdamW is fine here.

3. Do NOT worry about the FP8 base weights. They're frozen. The optimizer
   never touches them. The backward pass through the FP8 base linear uses
   dequantized bf16 for the gradient computation, and the LoRA gradient is
   standard autograd on bf16 Parameters. Muon sees normal bf16 gradients.


## 6. Summary Table

| Dimension | AdamW (current) | Muon (proposed) | Riemannion (future) |
|-----------|----------------|-----------------|---------------------|
| Optimizer state | 8 B/param (fp32 m+v) | 2 B/param (bf16 mom) | ~6 B/param |
| LR range | 1e-4 | ~0.02 (or KiMuon ~3e-4) | Similar to Muon |
| Spectral property | Per-element adaptive | Uniform spectral update | Manifold-aware |
| LoRA factorization aware | No | No (but provably OK) | Yes |
| Implementation effort | Done | 2-line change | Custom optimizer |
| NS overhead | N/A | <0.001% FLOPS | <0.01% FLOPS |
| Empirical evidence | Baseline | Uniform spectral growth paper | +1-2% on benchmarks |

The recommendation is to try Option A (torch.optim.Muon on LoRA params) in
the next training run. If it shows improvement, consider Option C (KiMuon) for
LR compatibility. Riemannion is the theoretically correct approach but should
wait until we have evidence that naive Muon improves over Adam.


## Sources

- [Keller Jordan: Muon blog post](https://kellerjordan.github.io/posts/muon/)
- [KellerJordan/Muon reference implementation](https://github.com/KellerJordan/Muon/blob/master/muon.py)
- [torch.optim.Muon documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html)
- [LoRA meets Riemannion (arXiv:2507.12142)](https://arxiv.org/abs/2507.12142)
- [Uniform Spectral Growth of Muon in LoRA (arXiv:2602.06385)](https://arxiv.org/abs/2602.06385)
- [Effective Quantization of Muon Optimizer States (arXiv:2509.23106)](https://arxiv.org/abs/2509.23106)
- [Jeremy Bernstein: Deriving Muon](https://jeremybernste.in/writing/deriving-muon)
- [NVIDIA NeMo Emerging Optimizers: Muon](https://docs.nvidia.com/nemo/emerging-optimizers/latest/apidocs/orthogonalized-optimizers.html)

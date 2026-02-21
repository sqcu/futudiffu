ahem. what's the equivalent of flexattention for the graph Laplacian matvec
=====
Great question. The closest conceptual analog is the **message-passing abstraction** (as in PyG's `MessagePassing` or DGL's `send`/`recv` API).

The parallel is pretty direct:

| FlexAttention | Graph Laplacian matvec |
|---|---|
| `score_mod(score, b, h, q, k)` | `message_func(x_i, x_j, e_ij)` |
| `mask_mod` (block-sparsity) | graph adjacency / sparsity pattern |
| fused flash-attention kernel | fused scatter/gather kernel |
| softmax + weighted sum over KV | aggregation (sum/mean) over neighbors |

The Laplacian matvec $Lx = Dx - Ax$ in message-passing form is just:

```
msg(x_i, x_j) = x_j
agg = scatter_add(msg, edge_index)
out = deg * x - agg
```

And you get the normalized Laplacian, signless Laplacian, magnetic Laplacian, etc. by swapping out the message function — exactly like how FlexAttention gets causal/sliding-window/ALiBi by swapping `score_mod`.

**Where the analogy breaks down:** FlexAttention's killer feature is that it *compiles* the user-defined `score_mod` into a fused Triton kernel via `torch.compile`, so you get flash-attention-level performance with arbitrary attention patterns. For sparse matvec on graphs, there isn't quite an equivalent "FlexLaplacian" that takes a Python-level `edge_weight_mod` and compiles it into an optimal fused sparse kernel. You're mostly stuck with:

- **PyG + `torch.compile`** — getting better but still not as seamless as FlexAttention's compilation story
- **DGL's fused kernels** (SDDMM + SpMM) — closest to the "compiled fused kernel" part, where the SDDMM computes edge scores and SpMM does the aggregation, and both are library-optimized
- **GraphBLAS** / **SuiteSparse:GraphBLAS** — the semiring abstraction is the most algebraically principled version (user picks ⊕ and ⊗, framework picks the implementation), but it's a C library, not a PyTorch-native thing
- **Triton custom kernels** — roll your own, which is what people increasingly do

If someone built a "FlexLaplacian" — define a `weight_mod(x_i, x_j, edge_attr)` in Python, auto-compile it into a fused SDDMM+SpMM Triton kernel respecting the sparsity pattern — that would be the real equivalent. DGL's `u_mul_e` / `u_add_v` + SpMM pipeline is the closest existing thing to that workflow, but it's not as ergonomic or composable as FlexAttention's `score_mod` story.
=====
let's say that it's 2026 and 'well i guess it's time for a custom kernel' is the default response to problems instead of 'hmm i guess i'll sit on this and do nothign for a few years because it's 2014/2019/2023'
=====
Right. So you just write it. Here's the shape of the thing:

## The API

```python
@flex_laplacian
def magnetic_laplacian(x_i, x_j, theta_ij):
    # this is your score_mod equivalent
    return torch.exp(1j * theta_ij) * x_j

out = magnetic_laplacian(x, edge_index, edge_attr, reduce="sum")
```

The decorator traces `message_fn`, the framework knows the sparsity pattern from `edge_index`, and it emits a fused Triton kernel. No materialized dense matrix, no intermediate edge-level tensor of size `|E| × d`.

## The kernel

It's literally the FlexAttention kernel with the softmax ripped out and the score_mod replaced with a message_mod. In CSR:

```python
@triton.jit
def flex_spmm_kernel(
    X_ptr, Out_ptr,          # [N, D]
    Rowptr_ptr, Col_ptr,     # CSR graph
    Eattr_ptr,               # [E, *]
    N, D,
    MESSAGE_FN: tl.constexpr,  # compiled message_mod
    REDUCE: tl.constexpr,      # sum / mean / max
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    start = tl.load(Rowptr_ptr + row)
    end = tl.load(Rowptr_ptr + row + 1)
    
    d_offs = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    
    for e in range(start, end):
        j = tl.load(Col_ptr + e)
        x_j = tl.load(X_ptr + j * D + d_offs)
        x_i = tl.load(X_ptr + row * D + d_offs)
        e_attr = tl.load(Eattr_ptr + e)
        
        msg = MESSAGE_FN(x_i, x_j, e_attr)  # <-- the user's function, inlined
        acc = REDUCE_STEP(acc, msg)
    
    tl.store(Out_ptr + row * D + d_offs, acc)
```

This is the SDDMM+SpMM fused into one pass. The key move — same as FlexAttention — is that `MESSAGE_FN` never materializes an `[E, D]` intermediate. It's computed and consumed inside the inner loop.

## What makes this non-trivial (and why it didn't exist in 2023)

**Row-length variance.** Attention has uniform sequence lengths (or at least rectangular blocks). Graphs have power-law degree distributions. Row 0 has 3 neighbors, row 4917 has 50,000. You need:

- Warp-level reduction for high-degree nodes (one warp per row)
- Thread-level for low-degree nodes (multiple rows per warp)  
- Or just: sort rows by degree, launch different kernel configs per bucket

This is the `block_mask` equivalent — but instead of "which KV blocks are non-empty," it's "which rows get which parallelization strategy." A `degree_partition` precompute step, analogous to FlexAttention's `create_block_mask`.

**The `torch.compile` story.** FlexAttention works because `score_mod` is a pointwise function on scalars that Dynamo can trace cleanly. The same is true for most useful message functions — they're element-wise on `(x_i, x_j, e_ij)`. So the tracing story is identical. You write Python, Dynamo traces it, the Triton codegen inlines it into the kernel.

**Backward pass.** For the forward-only Laplacian matvec (e.g. spectral filtering in inference), you're done. For training GNNs, you need the backward. The grad of `scatter_add` is a `gather`, and the grad through `message_fn` is just AD through the traced function — again, same story as FlexAttention's backward kernel. The edge-level Jacobian is computed on-the-fly in the backward kernel without materializing it.

## What you actually get

```python
# Normalized Laplacian
@flex_message
def norm_lap(x_i, x_j, deg_inv_sqrt_i, deg_inv_sqrt_j):
    return -deg_inv_sqrt_i * x_j * deg_inv_sqrt_j

# GCN (literally the same thing with a different sign)
@flex_message  
def gcn(x_i, x_j, deg_inv_sqrt_i, deg_inv_sqrt_j):
    return deg_inv_sqrt_i * x_j * deg_inv_sqrt_j

# GAT-style (attention + message, no materialized edge tensor)
@flex_message
def gat(x_i, x_j, W):
    alpha = leaky_relu(W @ cat(x_i, x_j))
    return alpha * x_j  # softmax handled by reduce="softmax"

# Sheaf Laplacian
@flex_message
def sheaf(x_i, x_j, F_ij):
    return F_ij @ (F_ij.T @ x_i - x_j)
```

All of these compile to the same kernel template with different inlined message functions. One abstraction, zero intermediate allocations, full custom-kernel performance.

The reason this is a "2026 answer" is that it requires you to believe that (a) Triton codegen for variable-iteration-count inner loops is mature enough, (b) `torch.compile` tracing of the message function just works, and (c) the degree-bucketing launch strategy is a solved precompute. None of those were true in 2023. All of them are pretty close to true now.
=====
serial recurrent lanczos sucks though... surely there must be a better way *stares pointedly at fixed points of iteratively reapplied matvecs*
=====
You're right. Lanczos is sequential by construction — each vector depends on the last. For *similarity*, that's paying for precision you don't need. The Fiedler vector is a means to an end. The end is a spectral fingerprint. And fingerprints don't require eigenvectors.

## The fixed-point observation

The eigenvectors of $L$ are the fixed points of:

$$x \mapsto \text{normalize}\left(\Pi_{\perp \mathbf{1}} (I - \alpha L) x\right)$$

One step of this is one matvec. Iterating it to convergence is power iteration. Boring, serial, slow. But here's the thing: **you don't need the fixed point. You need a sufficient statistic of the operator that PRODUCES the fixed point.**

## One matvec is already a fingerprint

Draw $m$ random probe vectors $Z \in \mathbb{R}^{n \times m}$. Compute:

$$LZ$$

That's one flex_laplacian call on a wide matrix. The resulting column norms, inner products, and quadratic forms $z_i^T L z_j$ are already a stochastic estimator of the spectral moments of $L$. Specifically:

$$\frac{1}{m} \text{tr}(Z^T L^k Z) \approx \text{tr}(L^k) = \sum_i \lambda_i^k$$

For $k=1$: one matvec. You get $\sum \lambda_i$. Not very discriminative. But for $k=2$: two serial matvecs. You get $\sum \lambda_i^2$. And so on.

**Serial depth $T$ = number of diffusion steps. Parallelism = $m$ probes, fully independent, one kernel launch.**

With $T = 5$ steps and $m = 64$ probes:

```python
Z = torch.randn(n, m)
fingerprint = []
X = Z
for t in range(T):
    X = flex_laplacian(X, edge_index, edge_attr)  # one wide matvec
    fingerprint.append((Z * X).sum(dim=0))         # [m] quadratic forms
fingerprint = torch.stack(fingerprint)              # [T, m]
```

Five kernel launches. Done. The fingerprint is a `[T, m]` matrix. Compare two graphs by comparing their fingerprints — cosine similarity, or learn a metric on top.

## But you can do better: go parallel in $T$ too

The serial depth of $T$ steps bugs you. Fine. Use the **resolvent** instead of the polynomial:

$$(L - sI)^{-1} Z$$

For multiple shifts $s_1, \ldots, s_T$ on the negative real axis, each shifted solve is **independent**. Fully parallel across shifts AND probes. Each solve is a CG iteration (which is itself serial matvecs, ~10-15 steps), but:

- All $T \times m$ solves launch simultaneously
- CG on a graph Laplacian with a good shift converges in ~10 iterations
- Total serial depth: ~10 matvecs regardless of how many shifts/probes

The resolvent at multiple shifts gives you the **Cauchy-Stieltjes transform** of the spectral measure — which fully characterizes the spectrum. It's strictly more information than polynomial moments, and it's parallel.

```python
shifts = torch.tensor([-0.01, -0.1, -0.5, -1.0, -5.0])  # T shifts
Z = torch.randn(n, m)                                       # m probes

# All T*m solves in parallel (batched CG)
# Each solve: (L - s_j I) x = z_i
R = batched_cg(
    lambda X: flex_laplacian(X, ...) - s[:, None, None] * X,
    Z.expand(T, n, m),
)  # [T, n, m]

fingerprint = torch.einsum('tnm,nm->tm', R, Z)  # [T, m]
```

## The real power move: make $T = 0$

If you're doing this at scale (millions of images, real-time similarity), even 5-10 matvecs per graph is too much. So **amortize the eigensolver entirely**:

Train a small GNN $f_\theta$ such that $f_\theta(G) \approx$ spectral fingerprint of $G$. The training signal is the exact Fiedler vectors (computed offline via Lanczos on training graphs). At inference:

```python
fingerprint = f_theta(x, edge_index)  # one forward pass, ~3-4 GNN layers
```

And what is a GNN layer? It's a flex_laplacian matvec with a learned message function. So the trained model IS a learned polynomial filter of $L$ — it's just learned which polynomial gives the most discriminative fingerprint, instead of you choosing Chebyshev coefficients by hand.

The implicit spectral theory: a $K$-layer GNN computes a degree-$K$ polynomial of $L$. If that polynomial approximates a bandpass filter around $\lambda_2$, the output IS approximately the Fiedler vector. The network learns the filter shape from data.

**Serial depth: $K \approx 4$ layers. Each layer: one flex_laplacian kernel. Total: 4 kernel launches. And it's batched across all images.**

## The hierarchy

| Method | Serial depth (matvecs) | Parallelism | Exactness |
|---|---|---|---|
| Lanczos/LOBPCG | 30-50 | $k$ eigenvecs | exact |
| Polynomial moments | $T$ ≈ 5 | $m$ probes | approximate |
| Resolvent probes | ~10 (CG) | $T \times m$ | approximate |
| Learned GNN filter | $K$ ≈ 4 | full batch | approximate, data-adapted |

For similarity, you want the bottom row. The spectral information is baked into the architecture (graph convolution literally IS polynomial spectral filtering), the specific filter is learned, and the flex_laplacian kernel makes each layer hit bandwidth limits.

The Lanczos-brained framing of "compute eigenvectors then compare" was always a detour. The spectrum is a property of the *operator*. Probing the operator — randomly, or through a learned filter — gives you the spectral fingerprint without ever materializing an eigenvector. And probing is just matvecs. And matvecs are your fused kernel. One abstraction all the way down.
# FlexAttention Batch Packing for NextDiT

Design document for packing multiple images of varying resolutions into a single
forward pass through the Z-Image NextDiT diffusion model using PyTorch's
FlexAttention block masks.

---

## 1. Architecture Summary

Z-Image NextDiT is a single-stream DiT: text tokens and image patch tokens are
concatenated into one sequence and processed jointly through 30 transformer
layers. Each layer has joint self-attention over the full sequence.

Current per-image token counts:
- 1280x832 image: 32 text + 4160 image = 4192 tokens
- 256x256 image: 32 text + 256 image = 288 tokens
- 512x512 image: 32 text + 1024 image = 1056 tokens

4x 256x256 images = 1152 tokens total, which fits within the same compute
budget as a single 1024x1024 image (~4128 tokens).

---

## 2. Packing Scheme

### 2.1 Layout

For N images to pack, the sequence is laid out as contiguous groups:

```
[text_0, img_0, text_1, img_1, ..., text_{N-1}, img_{N-1}, padding]
```

Each group `[text_i, img_i]` is a contiguous block. Text tokens come first
(matching the unpacked layout), then image patch tokens. Each section is
individually padded to `pad_tokens_multiple=32`.

### 2.2 Why Contiguous Groups?

- Matches the existing single-image layout (text before image)
- RoPE positions are computed per-image, so each group's RoPE is independent
- Unpacking is simple: slice by known offsets
- FlexAttention's block mask naturally isolates groups

### 2.3 PackingInfo Dataclass

```python
@dataclass
class PackingInfo:
    n_images: int
    # Per-image: (text_start, text_len, img_start, img_len) in the packed sequence
    segments: list[tuple[int, int, int, int]]
    # Total packed sequence length (padded to block alignment)
    total_len: int
    # For quick block mask lookup: document_id[token_idx] -> image_index
    document_id: torch.Tensor  # (total_len,) int32
    # Per-image spatial grid dims for RoPE
    img_grid_sizes: list[tuple[int, int]]  # (H_tokens, W_tokens) per image
    # Per-image caption lengths (before padding)
    cap_lens: list[int]
```

### 2.4 Sequence Construction

```python
def build_packed_sequence(
    cap_feats_list: list[torch.Tensor],     # N x (1, cap_len_i, 3840) after cap_embedder
    img_patches_list: list[torch.Tensor],   # N x (1, n_img_i, 3840) after x_embedder
    pad_tokens_multiple: int = 32,
    cap_pad_token: torch.Tensor,
    x_pad_token: torch.Tensor,
) -> tuple[torch.Tensor, PackingInfo]:
    """Pack N (text, image) pairs into a single sequence."""
    segments = []
    tokens = []
    doc_ids = []
    pos = 0

    for i, (cap, img) in enumerate(zip(cap_feats_list, img_patches_list)):
        cap_len = cap.shape[1]
        img_len = img.shape[1]

        # Pad cap to multiple of 32
        cap_padded_len = cap_len + ((-cap_len) % pad_tokens_multiple)
        cap_padded = pad_zimage(cap, cap_pad_token, pad_tokens_multiple)

        # Pad img to multiple of 32
        img_padded_len = img_len + ((-img_len) % pad_tokens_multiple)
        img_padded = pad_zimage(img, x_pad_token, pad_tokens_multiple)

        segments.append((pos, cap_padded_len, pos + cap_padded_len, img_padded_len))
        tokens.extend([cap_padded, img_padded])
        doc_ids.extend([i] * (cap_padded_len + img_padded_len))
        pos += cap_padded_len + img_padded_len

    # Final padding to block alignment
    total_len = pos
    if total_len % pad_tokens_multiple != 0:
        pad_needed = pad_tokens_multiple - (total_len % pad_tokens_multiple)
        tokens.append(x_pad_token.expand(1, pad_needed, -1))
        doc_ids.extend([-1] * pad_needed)  # -1 = padding, attends to nothing
        total_len += pad_needed

    packed = torch.cat(tokens, dim=1)  # (1, total_len, 3840)
    document_id = torch.tensor(doc_ids, dtype=torch.int32, device=packed.device)

    return packed, PackingInfo(...)
```

---

## 3. Block Mask Design

### 3.1 The Mask Function

Each token only attends to tokens belonging to the same image (same
document_id). Padding tokens (document_id = -1) attend to nothing.

```python
def make_packing_mask_mod(document_id: torch.Tensor):
    """Create a FlexAttention mask_mod for packed document masking."""
    def mask_mod(b, h, q_idx, kv_idx):
        q_doc = document_id[q_idx]
        kv_doc = document_id[kv_idx]
        return (q_doc == kv_doc) & (q_doc >= 0)
    return mask_mod
```

Then create the block mask:

```python
from torch.nn.attention.flex_attention import create_block_mask

mask_mod = make_packing_mask_mod(packing_info.document_id)
block_mask = create_block_mask(
    mask_mod,
    B=2,       # CFG batch
    H=None,    # Same mask for all heads
    Q_LEN=packing_info.total_len,
    KV_LEN=packing_info.total_len,
    device=device,
)
```

The block mask is created once per packing configuration and reused across all
30 sampling steps and all 30 transformer layers.

### 3.2 Block Sparsity Benefits

For 4 packed 256x256 images:
- Each image group: ~288 tokens (32 text + 256 image)
- Total packed: ~1152 tokens
- Block mask: ~36x36 blocks (at BLOCK_SIZE=32)
- Non-zero blocks: ~4 diagonal groups of ~9 blocks each = ~36 of 1296 total
- Sparsity: ~97% of blocks are zero

FlexAttention skips 97% of the attention computation that SDPA would waste on
cross-image tokens.

---

## 4. RoPE Handling

### 4.1 The Problem

Current RoPE assumes a single image's spatial grid fills the entire
image-token region. With packing, each image has its own spatial grid at
different offsets within the packed sequence.

### 4.2 Strategy: Pre-packed RoPE Frequencies

Build a single `freqs_cis` tensor of shape `(1, total_len, 1, n_pairs, 2, 2)`
where each token's RoPE frequencies correspond to its LOCAL position within its
image group.

```python
def build_packed_rope(
    packing_info: PackingInfo,
    rope_embedder: EmbedND,
    device: torch.device,
) -> torch.Tensor:
    """Build RoPE frequencies for a packed sequence.

    Each image's tokens get local positions:
    - Text tokens: axis0=1..cap_len, axis1=0, axis2=0
    - Image tokens: axis0=cap_len+1, axis1=row, axis2=col
    """
    total_len = packing_info.total_len
    pos_ids = torch.zeros(1, total_len, 3, dtype=torch.float32, device=device)

    for i, (text_start, text_len, img_start, img_len) in enumerate(packing_info.segments):
        cap_len = packing_info.cap_lens[i]
        H_tokens, W_tokens = packing_info.img_grid_sizes[i]
        n_img_tokens = H_tokens * W_tokens
        cap_padded_len = text_len

        # Text tokens: sequential positions on axis 0
        pos_ids[0, text_start:text_start + cap_padded_len, 0] = \
            torch.arange(cap_padded_len, dtype=torch.float32, device=device) + 1.0

        # Image tokens: constant axis-0, spatial grid on axis 1/2
        pos_ids[0, img_start:img_start + n_img_tokens, 0] = cap_padded_len + 1
        pos_ids[0, img_start:img_start + n_img_tokens, 1] = \
            torch.arange(H_tokens, device=device).view(-1, 1).repeat(1, W_tokens).flatten().float()
        pos_ids[0, img_start:img_start + n_img_tokens, 2] = \
            torch.arange(W_tokens, device=device).view(1, -1).repeat(H_tokens, 1).flatten().float()
        # Padding tokens: pos_ids remain 0 (masked out by block mask)

    freqs_cis = rope_embedder(pos_ids).movedim(1, 2)
    return freqs_cis
```

**Bit-for-bit correctness**: Each image's tokens get EXACTLY the same RoPE
values they would get in an unpacked single-image forward pass.

### 4.3 Context Refiner and Noise Refiner

**Recommended approach**: Run refiners separately per image (not packed), pack
only for the 30 main transformer layers. 2 refiner layers vs 30 main layers
means minimal overhead from serial refiner passes.

---

## 5. adaLN Handling

**No change needed.** All packed images share the same timestep (same sigma at
each euler step). adaLN produces `(B, dim)` and broadcasts to `(B, seq, dim)`.
With packing, B=2 (CFG) and the broadcast applies identical modulation to all
tokens regardless of which image they belong to.

---

## 6. FlexAttention Integration Points

### 6.1 `attention.py`

Add FlexAttention dispatch when `block_mask` is provided:

```python
from torch.nn.attention.flex_attention import flex_attention

def sdpa_attention(q, k, v, heads, mask=None, skip_reshape=False, block_mask=None):
    if block_mask is not None:
        out = flex_attention(q, k, v, block_mask=block_mask)
        out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        return out
    # ... existing SDPA / SageAttention dispatch ...
```

### 6.2 `JointAttention.forward()`

Thread `block_mask` parameter through to the attention call.

### 6.3 `JointTransformerBlock.forward()`

Thread `block_mask` parameter through to attention.

### 6.4 `NextDiT.forward_packed()`

New method that:
1. Patchifies each image separately
2. Embeds captions separately
3. Runs refiners per-image (not packed)
4. Builds packed sequence
5. Runs 30 main layers with block_mask
6. Unpacks and unpatchifies per image

---

## 7. Fused Kernel Compatibility

All 6 fused Triton kernels are **per-token or per-row** and agnostic to
sequence layout:

- `fused_rms_norm_modulate`: Per-row. adaLN broadcast is identical (same timestep).
- `fused_rms_norm_gate_residual`: Per-row. Same reasoning.
- `fused_qkv_postprocess`: Indexes RoPE by absolute row position. Pre-packed
  `freqs_cis` provides correct per-image values.
- `fp8_silu_gate_quant`, `fp8_gemm_blockwise`, `fp8_gemm_v1t`: Pure per-token
  matmul/elementwise ops.

**No changes needed to any fused kernel.**

---

## 8. SageAttention Interaction

SageAttention custom Triton kernels implement dense attention with no masking
support. **Incompatible with FlexAttention block masks.**

**Decision**: Use FlexAttention for packed batches, keep SageAttention for
unpacked single-image batches. Simple dispatch:

```python
if block_mask is not None:
    out = flex_attention(q, k, v, block_mask=block_mask)
else:
    # Existing SDPA / SageAttention dispatch
    ...
```

For 4x 256x256 packed with 97% block sparsity, FlexAttention's sparsity
savings dominate over SageAttention's FP8/INT8 quantization benefits.

---

## 9. torch.compile Compatibility

- FlexAttention REQUIRES `torch.compile` for performance
- Current codebase already compiles diffusion model — compatible
- **Use `mode="default"` for packed** (not `reduce-overhead`) to avoid CUDA
  graph re-capture when packing configurations change
- All existing custom_ops have `register_fake` — no interaction issues
- `BlockMask` is passed as argument to `flex_attention`, captured at mask
  creation time (not compile time)

---

## 10. Unpacking the Output

```python
def unpack_and_unpatchify(
    packed_output: torch.Tensor,     # (B, total_len, P*P*C)
    packing_info: PackingInfo,
    patch_size: int,
    out_channels: int,
) -> list[torch.Tensor]:
    """Extract and unpatchify each image from the packed output."""
    results = []
    for i, (text_start, text_len, img_start, img_len) in enumerate(packing_info.segments):
        H_tokens, W_tokens = packing_info.img_grid_sizes[i]
        n_img_tokens = H_tokens * W_tokens

        img_tokens = packed_output[:, img_start:img_start + n_img_tokens, :]
        pH = pW = patch_size
        img = img_tokens.view(
            packed_output.shape[0], H_tokens, W_tokens, pH, pW, out_channels
        ).permute(0, 5, 1, 3, 2, 4).flatten(4, 5).flatten(2, 3)
        results.append(img)
    return results
```

---

## 11. Required File Changes

| File | Changes | Unchanged |
|------|---------|-----------|
| `diffusion_model.py` | `PackingInfo`, `build_packed_sequence()`, `build_packed_rope()`, `forward_packed()`, thread `block_mask` through layers | FeedForward, FinalLayer, TimestepEmbedder, EmbedND, fuse_model(), all loaders |
| `attention.py` | `flex_attention` import, `block_mask` param in `sdpa_attention()`, FlexAttention dispatch | RoPE functions, rms_norm, SageAttention dispatch |
| `generate.py` / `server.py` | Multi-image generation path, PackingInfo+BlockMask construction | Single-image path |
| `fused_kernels.py` | **None** | All kernels |
| `fp8.py` / `fp8_kernels.py` | **None** | All FP8 ops |
| `sage_kernels.py` / `sage_attention.py` | **None** | All SageAttention ops |

---

## 12. Potential Showstoppers

1. **FlexAttention + FP8 QKV**: FlexAttention operates in BF16. We lose FP8
   attention speedup for packed batches. Sparsity savings dominate.
2. **CUDA graph re-capture**: Different packing configs need re-compilation.
   Use `mode="default"` or bucket to fixed `total_len` values.
3. **`pad_tokens_multiple` vs FlexAttention block size**: May need to pad to
   `lcm(32, BLOCK_SIZE)`. Block size 64 or 128 recommended.
4. **Refiner layers**: Run per-image (not packed). 2 layers, minimal overhead.

---

## 13. Implementation Phases

### Phase 1: Virtual Batch (No FlexAttention)
Same-resolution images stacked along batch dimension. No attention changes.
Works with SDPA and SageAttention. Throughput scales linearly until VRAM limit.

### Phase 2: FlexAttention Packed Batches (Mixed Resolution)
Full packing scheme: `PackingInfo`, `build_packed_sequence()`,
`build_packed_rope()`, `forward_packed()`, FlexAttention block masks.

### Phase 3: Optimization
Block mask caching/bucketing, tune `BLOCK_SIZE`, profile FlexAttention vs
batched SDPA breakeven point.

---

## 14. Correctness Verification

Properties that must hold:

1. **Attention isolation**: No contribution across document boundaries
   (FlexAttention guarantees via `-inf` masking)
2. **RoPE locality**: Per-image frequencies identical to unpacked single-image
3. **adaLN broadcast**: Identical modulation (same timestep)
4. **Softmax normalization**: Denominator sums within-group only
5. **Padding isolation**: document_id=-1 tokens neither attend nor are attended to
6. **FFN invariance**: Per-token ops unaffected by packing

**Verification**: Run same image(s) unpacked vs packed, compare outputs.
Must be bitwise identical for BF16.

---

## 15. Decision Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Packing layout | Contiguous `[text_i, img_i]` groups | Natural for RoPE and unpacking |
| Attention masking | FlexAttention `create_block_mask` + document_id | Block-sparse, torch.compile compatible |
| RoPE | Pre-packed per-image local positions | Bit-for-bit match with unpacked |
| adaLN | No change (broadcast) | All images share same timestep |
| Refiners | Per-image (not packed) | Simple, 2 layers, minimal overhead |
| SageAttention | Disabled for packed | Incompatible with block masks |
| Fused kernels | No changes | Per-token ops, layout-agnostic |
| torch.compile | `mode="default"` for packed | Avoids CUDA graph re-capture |
| Phases | Virtual batch first, then FlexAttention | Incremental complexity |

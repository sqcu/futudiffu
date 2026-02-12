# Memorandum: Spec Gaps Not in CLAUDE.md

Supplementary context for the futudiffu agent. Everything here was in the original
spec but didn't make it into CLAUDE.md. Read this before claiming validation is done.

---

## 1. The Exact Prompt Text

This is the literal string from the workflow JSON (node 6, `widgets_values[0]`).
It must be reproduced character-for-character, including the newlines and punctuation:

```
ahem.
*ting ting ting ting ting*
the query model for this is a LARGE LANGUAGE MODEL, specifically QWEN-3-4B, a GENERAL PURPOSE SEMANTIC PARSER which is able to WRITE SENTENCES AT A TIME when they are participating in dialogue. however, in this situation, they are being used as a hidden state generator to steer an *image generation model*, z-image.

qwen-3-4b, draw me an "enormous laser shark for the sega saturn".
```

The negative prompt is the empty string `""`.

---

## 2. The Multiplier Chain (Three-Stage Rescaling)

This is the most likely source of silent numerical bugs. There are three different
"multiplier" values that interact:

**Config level**: `GenerateConfig.multiplier = 1.0` (this is the ModelSamplingAuraFlow override)

**Sigma table construction** (in `generate.py`):
```python
sigma_table = build_sigmas(shift=1.0, multiplier=config.multiplier * 1000)
# i.e., build_sigmas(multiplier=1000.0)
```
Inside `build_sigmas`: `ts = arange(1,1001)/1000 * 1000` then `sigmas = time_snr_shift(1.0, ts/1000)`.
The `*1000` and `/1000` cancel. Result: `sigmas = arange(1,1001)/1000` (linear ramp 0.001..1.0).

**Timestep passed to model** (in `generate.py:model_fn`):
```python
timestep = sigma * config.multiplier  # = sigma * 1.0 = sigma
```
So the model receives sigma directly as its timestep (a value in [0, 1]).

**Inside the diffusion model forward** (`diffusion_model.py`):
```python
t = 1.0 - timesteps        # inverts: sigma=0.967 → t=0.033
t_embed = self.t_embedder(t * self.time_scale)  # t * 1000.0
```

The chain is: `sigma` → `timestep=sigma*1.0` → `t=1-sigma` → `t*1000` → `timestep_embedding`.
If any of these stages is wrong, the model output is garbage but might not crash.

---

## 3. comfy_kitchen: The FP8 Loading Layer

CLAUDE.md mentions FP8 but omits the intermediate abstraction layer. ComfyUI's FP8
loading goes through `comfy_kitchen`, not directly through raw safetensors:

- `comfy_kitchen/tensor/base.py` — `QuantizedTensor` class: dispatches operations
  to layout-specific implementations. When a `nn.Linear` has a QuantizedTensor as
  its weight, forward() calls `QuantizedTensor.__torch_function__` which intercepts
  `F.linear` and routes to the layout's matmul kernel.

- `comfy_kitchen/tensor/fp8.py` — `BlockWiseFP8Layout`: stores `float8_e4m3fn` data
  tensor + `float32` scale tensor with shape `(ceil(out/block_size), ceil(in/block_size))`.
  Its matmul method calls the Triton FP8 GEMM kernel.

- `comfy_kitchen` **disables Triton globally**: `ck.registry.disable("triton")` in
  `quant_ops.py:24`. This means ComfyUI's native FP8 path does NOT use Triton — it
  uses PyTorch's built-in FP8 ops instead. Only the QuantOps custom node re-enables
  Triton for blockwise.

The futudiffu `fp8.py` FP8Linear class replaces this entire stack. Verify that:
1. The quantization of activations matches comfy_kitchen's per-block quantization
2. The scale tensor layout matches (row-major blocks of `block_size`)
3. The GEMM output dtype is bf16 (not fp16 — this was a bug in early versions)

---

## 4. Weight Detection Logic from State Dicts

ComfyUI auto-detects model architecture from state dict keys. If weight loading
fails, these are the detection paths to trace:

**Z-Image diffusion model** (`comfy/model_detection.py:438`):
- `dim == 3840` triggers Z-Image config
- n_layers: counted by `count_blocks(state_dict, "layers.")` — counts how many
  `layers.N.` prefixes exist
- cap_feat_dim: `state_dict["cap_embedder.1.weight"].shape[1]`
- Presence of `cap_pad_token` in state dict → `pad_tokens_multiple = 32`

**Qwen3-4B text encoder** (`comfy/sd.py:detect_te_model`):
- `post_attention_layernorm.weight.shape[0] == 2560` → Qwen3_4B

**VAE** (`comfy/sd.py`):
- Presence of `decoder.conv_in.weight` → AutoencoderKL
- `latent_channels = state_dict["decoder.conv_in.weight"].shape[1]`
- Absence of `encoder.` keys → decode-only is fine

**State dict key prefixes to strip**:
- Diffusion: `model.diffusion_model.` or `diffusion_model.` → strip
- Text encoder: `text_encoders.qwen25.transformer.` or `transformer.` → strip
- VAE: keys may be bare (no prefix)

---

## 5. Design Decisions Not Documented

These substitutions are intentional, not bugs:

- **`comfy.ops` replaced with plain `nn.Module`**: ComfyUI's `comfy.ops` wraps every
  `nn.Linear`, `nn.Conv2d`, etc. with dtype/device management and quantization dispatch.
  futudiffu replaces this with standard PyTorch modules + explicit dtype casting.
  `comfy/ops.py:604` hard-raises on quant formats not in `{float8_e4m3fn, float8_e5m2, nvfp4}`.

- **`comfy.model_management` replaced with explicit `.to(device)`**: ComfyUI has a
  centralized memory manager that tracks VRAM, handles offloading, etc. futudiffu
  just does `.to(device)` and `.to(dtype)` directly.

- **`custom_operations` pattern eliminated**: ComfyUI's `custom_operations` provides
  exactly one slot per model for intercepting ops. futudiffu's FP8Linear replaces
  this directly at the module level.

---

## 6. ComfyUI Source Line Numbers

For tracing exact code paths against the original:

| What | File | Lines |
|---|---|---|
| NextDiT class | `comfy/ldm/lumina/model.py` | 423 |
| NextDiT.forward | `comfy/ldm/lumina/model.py` | 810-859 |
| Z-Image detection | `comfy/model_detection.py` | 438-456 |
| ZImage config class | `comfy/supported_models.py` | 1082 |
| CONST model type | `comfy/model_sampling.py` | 62-76 |
| ModelSamplingDiscreteFlow | `comfy/model_sampling.py` | 249 |
| time_snr_shift | `comfy/model_sampling.py` | 244 |
| simple_scheduler | `comfy/samplers.py` | 405 |
| sample_euler | `comfy/k_diffusion/sampling.py` | 218 |
| Qwen3_4BConfig | `comfy/text_encoders/llama.py` | 177 |
| ZImageTEModel | `comfy/text_encoders/z_image.py` | top |
| chat template wrapping | `comfy/text_encoders/z_image.py` | ZImageTokenizer class |
| EmptySD3LatentImage | `comfy_extras/nodes_sd3.py` | 57 |
| Flux latent format | `comfy/latent_formats.py` | 153 |
| pad_to_patch_size | `comfy/ldm/common_dit.py` | full file |
| ops.py quant hard-raise | `comfy/ops.py` | 604 |

All paths relative to `/mnt/f/dox/ai/comfyui/ComfyUI/`.

---

## 7. Bugs Fixed in fp8_kernels.py

The `fp8_kernels.py` in this repo is copied from `/mnt/f/dox/ai/comfyui/QuantOps-reference/`,
which has fixes applied on top of the original ComfyUI-QuantOps. If you compare
against upstream or ComfyUI's installed copy, these are the differences:

1. **`other=0` → `other=0.0`** in all `tl.load(..., other=)` calls. Triton 3.x is
   strict about type: integer `0` fails type checking for float tensor loads.

2. **Hardcoded grid `cdiv(M, 128)` → `grid(META)` lambda**. The original hardcoded
   `BLOCK_SIZE_M=128` in the grid calculation, but the autotuner may select different
   block sizes. The fix uses `META["BLOCK_SIZE_M"]` to read the autotuned value.

3. **Native FP8 dot product**: `tl.dot(a_fp8, b_fp8, out_dtype=tl.float32)` — uses
   the hardware tensor cores directly instead of upcasting both operands to FP32
   before the dot product. SM 8.9 (Ada) has native FP8 tensor core support.

4. **Output dtype**: GEMM output respects model dtype (bf16), not hardcoded fp16.

---

## 8. CFGGuider / calc_cond_batch Conditioning Format

ComfyUI's conditioning is NOT just a tensor — it's a list of tuples:
```python
cond = [[hidden_states_tensor, {"pooled_output": pooled_tensor}]]
```

For Z-Image/Lumina2:
- The hidden states tensor shape is `(B, seq_len, 2560)`
- There is no pooled output (the dict may be empty or absent)
- The conditioning list has exactly one entry (no multi-conditioning)

The `calc_cond_batch` function in `comfy/samplers.py` unpacks this format, handles
batching of multiple conditions, and applies area/mask conditioning. For our
single-prompt case, it simplifies to: just pass the hidden states tensor directly.

futudiffu's `generate.py` does this correctly (passes `positive_cond` / `negative_cond`
directly as tensors), but if debugging against ComfyUI's intermediate values, be
aware that ComfyUI wraps them in this list-of-tuples format.

---

## 9. Workflow Graph Topology

From the workflow JSON, the active execution graph is:

```
QuantizedUNETLoader (node 33: z_image_fp8_blockwise, float8_e4m3fn_blockwise, triton)
  → LoraLoader (node 31: mode=4 BYPASSED, so passthrough)
    → ModelSamplingAuraFlow (node 11: shift=1.0)
      → KSampler (node 3: seed=91849188298864, fixed, 30 steps, cfg=4, euler, simple, denoise=1)

CLIPLoader (node 18: qwen_3_4b.safetensors, type=lumina2)
  → CLIPTextEncode positive (node 6: the prompt text)
  → CLIPTextEncode negative (node 7: "")

ResolutionMaster (node 28: 1280x832)
  → EmptySD3LatentImage (node 13)

VAELoader (node 17: zimage.safetensors) → VAEDecode (node 8) → SaveImage (node 9)
```

Note: node 16 (UNETLoader) and node 34 (QuantizedCLIPLoader) exist in the workflow
but have **no output connections** — their outputs arrays are empty `[]`. They are
inactive/disconnected nodes. The active diffusion model loader is node 33
(QuantizedUNETLoader) and the active text encoder is node 18 (CLIPLoader with BF16).

For Config A (defective), you would reconnect nodes 6 and 7 to node 34's CLIP output
instead of node 18's.

---

## 10. What "Tests Pass" Actually Means

If the other agent claims "all tests pass," verify WHAT tests:

- **Import checks** (`bootstrap.py check`): Only proves modules parse and import
  without errors. Does NOT test any computation. This is what was verified before
  handoff.

- **Sigma schedule** (`bootstrap.py test-sampling`): Tests `build_sigmas` and
  `simple_scheduler` output. Pure CPU math. Was written but never actually run
  before handoff.

- **What has NOT been tested at all**:
  - torch CUDA availability (it's CPU-only, `cuda=False`)
  - Any model weight loading from safetensors files
  - Any forward pass through any neural network
  - Any comparison against ComfyUI reference tensors
  - Noise generation matching (CPU vs CUDA randn)
  - FP8 quantization / dequantization
  - Tokenizer output matching
  - VAE decode

Tests that "pass" without CUDA are not validating the spec. The spec requires
float-for-float CUDA tensor matching.

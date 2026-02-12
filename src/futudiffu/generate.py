"""End-to-end generation pipeline: exact ComfyUI workflow replication.

Functional signature:
    (model_paths, prompt, negative_prompt, seed, steps, cfg, width, height, ...) -> image_tensor

Pipeline steps:
1. Load text encoder -> tokenize prompt with chat template -> encode -> hidden states (B, seq, 2560)
2. Load text encoder -> tokenize "" -> encode -> hidden states (negative)
3. Create empty latent: zeros(1, 16, height//8, width//8)
4. Generate noise: torch.randn with manual_seed(seed)
5. Compute sigmas: simple_scheduler over ModelSamplingDiscreteFlow (shift=1.0), 30 steps + 0.0
6. CONST noise_scaling: x = sigma * noise + (1 - sigma) * latent
7. Euler loop x steps:
   - CFG: denoised = uncond + (cond - uncond) * cfg_scale
   - d = (x - denoised) / sigma
   - dt = sigma_next - sigma
   - x = x + d * dt
8. CONST inverse_noise_scaling: x / (1 - sigma_last)
9. Flux process_out: (latent / 0.3611) + 0.1159
10. VAE decode -> image tensor
11. Clamp to [0, 255] uint8

Two execution paths:
  comfyui_compat=True:  Validated ComfyUI-matching dtype flow (f16 TE, f32 loop, CPU noise).
  comfyui_compat=False: Performance path (unified bf16, CFG batching, torch.compile).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .sampling import (
    build_sigmas,
    const_calculate_denoised,
    const_inverse_noise_scaling,
    const_noise_scaling,
    flux_process_out,
    sample_euler,
    simple_scheduler,
    to_d,
)


@dataclass
class GenerateConfig:
    """Configuration for a single generation run."""
    # Model paths
    diffusion_model_path: str = ""
    text_encoder_path: str = ""
    vae_path: str = ""
    tokenizer_path: str | None = None

    # Generation params
    prompt: str = ""
    negative_prompt: str = ""
    seed: int = 91849188298864
    steps: int = 30
    cfg: float = 4.0
    width: int = 1280
    height: int = 832

    # Sampling params
    sampling_shift: float = 1.0
    multiplier: float = 1.0
    denoise: float = 1.0
    sampler: str = "euler"
    scheduler: str = "simple"

    # Device/dtype
    device: str = "cuda"
    dtype: str = "bfloat16"

    # FP8 config
    fp8_diffusion: bool = False
    fp8_text_encoder: bool = False
    fp8_block_size: int = 128

    # Compat mode: reproduce validated ComfyUI-matching dtype flow exactly.
    # When True: float16 TE, float32 sampling loop, CPU noise, sequential CFG,
    # no torch.compile. When False: unified bf16, CFG batching, torch.compile.
    comfyui_compat: bool = False

    # Attention backend: "sdpa" (PyTorch), "sage" (FP8 QK^T), "auto" (sage with fallback)
    attention_backend: str = "sdpa"

    # SageAttention configuration (only applies when attention_backend != "sdpa")
    sage_smooth_k: bool = True          # K-smoothing: subtract per-head channel mean
    sage_qk_quant: str = "fp8"          # "fp8" or "int8" for QK^T matmul
    sage_pv_quant: str = "bf16"         # "bf16" or "fp8" for PV matmul


def _get_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def generate(config: GenerateConfig, emitter=None) -> torch.Tensor:
    """Run the full generation pipeline.

    Args:
        config: Generation configuration.
        emitter: Optional TensorEmitter for recording intermediate tensors.

    Returns:
        (B, H, W, 3) uint8 image tensor.
    """
    device = torch.device(config.device)
    dtype = _get_dtype(config.dtype)
    compat = config.comfyui_compat

    # TE dtype: compat uses float16 (ComfyUI hardcode), fast uses model dtype
    te_dtype = torch.float16 if compat else dtype

    # --- Step 1-2: Text encoding ---
    te_label = "float16 [compat]" if compat else config.dtype
    print(f"[1/6] Loading text encoder (dtype={te_label})...")
    from .text_encoder import create_tokenizer, encode_prompt, load_text_encoder

    tokenizer = create_tokenizer(config.tokenizer_path)
    te_model = load_text_encoder(config.text_encoder_path, device=device, dtype=te_dtype)

    if config.fp8_text_encoder:
        from .fp8 import replace_linear_with_fp8
        from safetensors.torch import load_file
        te_sd = load_file(config.text_encoder_path, device=str(device))
        # Strip "model." or "transformer." prefix to match module tree keys
        remapped = {}
        for k, v in te_sd.items():
            new_key = k
            if new_key.startswith("model."):
                new_key = new_key[len("model."):]
            elif new_key.startswith("transformer."):
                new_key = new_key[len("transformer."):]
            remapped[new_key] = v
        replace_linear_with_fp8(te_model, remapped, block_size=config.fp8_block_size, output_dtype=te_dtype)

    if not compat:
        te_model = torch.compile(te_model, mode="default")

    print("  Encoding positive prompt...")
    positive_cond = encode_prompt(te_model, tokenizer, config.prompt, device=device)
    print(f"  Positive: {positive_cond.shape}")
    if emitter is not None:
        emitter.emit("text_encoder_pos", positive_cond)

    print("  Encoding negative prompt...")
    negative_cond = encode_prompt(te_model, tokenizer, config.negative_prompt, device=device)
    print(f"  Negative: {negative_cond.shape}")
    if emitter is not None:
        emitter.emit("text_encoder_neg", negative_cond)

    # In compat mode, cast TE output (float16) to model dtype (bf16).
    # In fast mode, TE already outputs in model dtype — no cast needed.
    if compat:
        positive_cond = positive_cond.to(dtype)
        negative_cond = negative_cond.to(dtype)

    # Free text encoder memory
    del te_model
    torch.cuda.empty_cache()

    # --- Step 3: Create empty latent ---
    # Compat: float32 (ComfyUI default torch dtype). Fast: model dtype (bf16).
    print("[2/6] Creating empty latent...")
    latent_h = config.height // 8
    latent_w = config.width // 8
    loop_dtype = torch.float32 if compat else dtype
    latent = torch.zeros(1, 16, latent_h, latent_w, device=device, dtype=loop_dtype)

    # --- Step 4: Generate noise ---
    # Compat: CPU float32 with torch.manual_seed (matches ComfyUI comfy/sample.py:11,27).
    # Fast: CUDA model-dtype with device generator — no CPU->GPU transfer.
    print("[3/6] Generating noise...")
    if compat:
        generator = torch.manual_seed(config.seed)
        noise = torch.randn(1, 16, latent_h, latent_w, dtype=torch.float32, generator=generator, device="cpu")
        noise = noise.to(device=device)
    else:
        generator = torch.Generator(device=device).manual_seed(config.seed)
        noise = torch.randn(1, 16, latent_h, latent_w, dtype=dtype, generator=generator, device=device)
    if emitter is not None:
        emitter.emit("noise", noise)

    # --- Step 5: Compute sigmas ---
    print("[4/6] Computing sigma schedule...")
    sigma_table = build_sigmas(shift=config.sampling_shift, multiplier=config.multiplier * 1000)
    sigmas = simple_scheduler(sigma_table, config.steps)
    # Compat: float32 sigmas (ComfyUI default). Fast: model dtype.
    sigmas = sigmas.to(device=device, dtype=loop_dtype)
    print(f"  Sigmas: {sigmas.shape} range [{sigmas[0]:.6f}, {sigmas[-1]:.6f}]")
    if emitter is not None:
        emitter.emit("sigmas", sigmas)

    # --- Step 6: CONST noise scaling ---
    x = const_noise_scaling(sigmas[0], noise, latent)

    # --- Step 7: Load diffusion model and run euler loop ---
    print("[5/6] Loading diffusion model...")

    if config.fp8_diffusion:
        from .diffusion_model import (
            create_diffusion_model,
            _detect_cap_feat_dim,
            _detect_n_layers,
            _detect_qk_norm,
            _strip_diffusion_prefix,
        )
        from .fp8 import replace_linear_with_fp8
        from safetensors.torch import load_file

        # Load FP8 state dict once
        diff_sd = load_file(config.diffusion_model_path, device=str(device))
        remapped = _strip_diffusion_prefix(diff_sd)
        del diff_sd

        # Create model skeleton (meta device, no weight data)
        n_layers = _detect_n_layers(remapped.keys())
        cap_feat_dim = _detect_cap_feat_dim(remapped)
        qk_norm = _detect_qk_norm(remapped.keys())
        diff_model = create_diffusion_model(dtype=dtype, n_layers=n_layers, cap_feat_dim=cap_feat_dim, qk_norm=qk_norm)

        # Inject FP8 weights into the skeleton (replaces nn.Linear with FP8Linear
        # where the state dict has FP8 weights+scales, and assigns remaining
        # non-FP8 parameters via load_state_dict with assign=True)
        replace_linear_with_fp8(diff_model, remapped, block_size=config.fp8_block_size, output_dtype=dtype)

        # Load remaining non-FP8 parameters (norms, embeddings, pad tokens, etc.)
        # Filter out weight_scale (already in FP8Linear) and .comfy_quant metadata.
        remaining = {}
        for k, v in remapped.items():
            if k.endswith(".weight_scale") or k.endswith(".comfy_quant"):
                continue
            remaining[k] = v
        diff_model.load_state_dict(remaining, strict=False, assign=True)
        del remapped, remaining

        diff_model = diff_model.to(device)
        diff_model.eval()
    else:
        from .diffusion_model import load_diffusion_model
        diff_model = load_diffusion_model(config.diffusion_model_path, device=device, dtype=dtype)

    if not compat:
        diff_model = torch.compile(diff_model, mode="reduce-overhead")

    # --- Precompute RoPE cache (numerically identical, safe in both modes) ---
    padded_h = latent_h + ((-latent_h) % diff_model.patch_size)
    padded_w = latent_w + ((-latent_w) % diff_model.patch_size)

    if compat:
        # Compat: sequential CFG with separate rope caches per cap_len.
        num_tokens_pos = positive_cond.shape[1]
        num_tokens_neg = negative_cond.shape[1]
        rope_cache_pos = diff_model.prepare_rope_cache(padded_h, padded_w, num_tokens_pos, device)
        if num_tokens_neg == num_tokens_pos:
            rope_cache_neg = rope_cache_pos
        else:
            rope_cache_neg = diff_model.prepare_rope_cache(padded_h, padded_w, num_tokens_neg, device)

        def model_fn(x_in: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            """CFG wrapper: sequential pos/neg (ComfyUI-matching dtype flow)."""
            # Cast float32 x_in to model dtype for the forward pass.
            model_input = x_in.to(dtype)
            timestep = sigma * config.multiplier

            model_output_cond = diff_model(
                model_input, timestep, positive_cond,
                num_tokens=num_tokens_pos, rope_cache=rope_cache_pos,
            )
            # calculate_denoised uses the ORIGINAL float32 x_in, not the
            # bf16 model_input (matches ComfyUI's BaseModel.apply_model).
            denoised_cond = const_calculate_denoised(sigma, model_output_cond, x_in)

            if config.cfg == 1.0:
                return denoised_cond

            model_output_uncond = diff_model(
                model_input, timestep, negative_cond,
                num_tokens=num_tokens_neg, rope_cache=rope_cache_neg,
            )
            denoised_uncond = const_calculate_denoised(sigma, model_output_uncond, x_in)
            return denoised_uncond + (denoised_cond - denoised_uncond) * config.cfg
    else:
        # Fast: pad conditioning to same length, batch pos+neg into single forward.
        pos_len = positive_cond.shape[1]
        neg_len = negative_cond.shape[1]
        max_len = max(pos_len, neg_len)
        if pos_len < max_len:
            positive_cond = F.pad(positive_cond, (0, 0, 0, max_len - pos_len))
        if neg_len < max_len:
            negative_cond = F.pad(negative_cond, (0, 0, 0, max_len - neg_len))

        num_tokens = max_len
        rope_cache = diff_model.prepare_rope_cache(padded_h, padded_w, num_tokens, device)

        if config.cfg != 1.0:
            cond_batch = torch.cat([positive_cond, negative_cond], dim=0)  # (2, seq, dim)

        def model_fn(x_in: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            """CFG wrapper: batched pos+neg in single forward pass."""
            timestep = sigma * config.multiplier

            if config.cfg == 1.0:
                output = diff_model(
                    x_in, timestep, positive_cond,
                    num_tokens=num_tokens, rope_cache=rope_cache,
                )
                return const_calculate_denoised(sigma, output, x_in)

            x_batch = x_in.expand(2, -1, -1, -1)
            t_batch = timestep.expand(2)
            output_batch = diff_model(
                x_batch, t_batch, cond_batch,
                num_tokens=num_tokens, rope_cache=rope_cache,
            )
            output_cond, output_uncond = output_batch.chunk(2, dim=0)
            denoised_cond = const_calculate_denoised(sigma, output_cond, x_in)
            denoised_uncond = const_calculate_denoised(sigma, output_uncond, x_in)
            return denoised_uncond + (denoised_cond - denoised_uncond) * config.cfg

    # Set attention backend (sage only applies when not in compat mode)
    if not compat and config.attention_backend != "sdpa":
        from .attention import set_attention_backend
        from .sage_attention import configure_sage
        set_attention_backend(config.attention_backend)
        configure_sage(
            smooth_k=config.sage_smooth_k,
            qk_quant=config.sage_qk_quant,
            pv_quant=config.sage_pv_quant,
        )
        parts = [config.attention_backend]
        if config.sage_smooth_k:
            parts.append("smooth_k")
        parts.append(f"qk={config.sage_qk_quant}")
        parts.append(f"pv={config.sage_pv_quant}")
        print(f"  Attention backend: {', '.join(parts)}")

    print(f"  Running euler sampler for {config.steps} steps...")

    def progress_callback(info):
        i = info['i']
        sigma = info['sigma']
        if (i + 1) % 5 == 0 or i == 0:
            print(f"    Step {i+1}/{config.steps}, sigma={sigma:.6f}")

    if emitter is not None:
        from .tensor_stream import make_euler_callback
        euler_cb = make_euler_callback(emitter, existing_callback=progress_callback)
    else:
        euler_cb = progress_callback

    x = sample_euler(model_fn, x, sigmas, callback=euler_cb)

    # --- Step 8: CONST inverse noise scaling ---
    # sigma_last = 0.0, so this is x / (1 - 0) = x (no-op)
    x = const_inverse_noise_scaling(sigmas[-1], x)

    if emitter is not None:
        emitter.emit("final_latent", x)

    # Free diffusion model memory
    del diff_model
    torch.cuda.empty_cache()

    # --- Step 9-10: VAE decode ---
    # Compat: cast float32 loop output back to model dtype for VAE.
    # Fast: already in model dtype.
    if compat:
        x = x.to(dtype)
    print("[6/6] VAE decode...")
    from .vae import load_vae, vae_decode

    vae_model = load_vae(config.vae_path, device=device, dtype=dtype)
    if not compat:
        vae_model = torch.compile(vae_model, mode="default")
    image = vae_decode(vae_model, x)
    if emitter is not None:
        emitter.emit("vae_output", image)

    del vae_model
    torch.cuda.empty_cache()

    # --- Step 11: Convert to uint8 ---
    # image is (B, 3, H, W) in [0, 1]
    # ComfyUI does: movedim(1, -1) then np.clip(image * 255, 0, 255).astype(np.uint8)
    image = image.movedim(1, -1)  # (B, H, W, 3)
    image_np = np.clip(image.cpu().float().numpy() * 255, 0, 255).astype(np.uint8)

    print("Done!")
    return image_np


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="futudiffu: exact ComfyUI workflow replication")
    parser.add_argument("--diffusion-model", required=True, help="Path to diffusion model safetensors")
    parser.add_argument("--text-encoder", required=True, help="Path to text encoder safetensors")
    parser.add_argument("--vae", required=True, help="Path to VAE safetensors")
    parser.add_argument("--tokenizer", default=None, help="Path to tokenizer directory")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt text")
    parser.add_argument("--seed", type=int, default=91849188298864)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=832)
    parser.add_argument("--shift", type=float, default=1.0, help="Sampling shift (ModelSamplingAuraFlow)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--fp8-diffusion", action="store_true", help="Use FP8 blockwise for diffusion model")
    parser.add_argument("--fp8-text-encoder", action="store_true", help="Use FP8 blockwise for text encoder")
    parser.add_argument("--comfyui-compat", action="store_true",
                        help="Use validated ComfyUI-matching dtype flow (f16 TE, f32 loop, CPU noise)")
    parser.add_argument("--attention-backend", default="sdpa", choices=["sdpa", "sage", "auto"],
                        help="Attention backend: sdpa (PyTorch), sage (FP8 QK^T), auto (sage with fallback)")
    parser.add_argument("--sage-smooth-k", action=argparse.BooleanOptionalAction, default=True,
                        help="K-smoothing: subtract per-head channel mean (default: enabled)")
    parser.add_argument("--sage-qk-quant", default="fp8", choices=["fp8", "int8"],
                        help="QK^T quantization: fp8 (3-bit mantissa) or int8 (7-bit)")
    parser.add_argument("--sage-pv-quant", default="bf16", choices=["bf16", "fp8"],
                        help="PV quantization: bf16 (safe) or fp8 (2x PV throughput, two-level accum)")
    parser.add_argument("--output", default="output.png", help="Output image path")
    parser.add_argument("--record-to", default=None, help="Directory to record tensor stream for validation")

    args = parser.parse_args()

    config = GenerateConfig(
        diffusion_model_path=args.diffusion_model,
        text_encoder_path=args.text_encoder,
        vae_path=args.vae,
        tokenizer_path=args.tokenizer,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        steps=args.steps,
        cfg=args.cfg,
        width=args.width,
        height=args.height,
        sampling_shift=args.shift,
        device=args.device,
        dtype=args.dtype,
        fp8_diffusion=args.fp8_diffusion,
        fp8_text_encoder=args.fp8_text_encoder,
        comfyui_compat=args.comfyui_compat,
        attention_backend=args.attention_backend,
        sage_smooth_k=args.sage_smooth_k,
        sage_qk_quant=args.sage_qk_quant,
        sage_pv_quant=args.sage_pv_quant,
    )

    emitter = None
    if args.record_to:
        from .tensor_stream import TensorEmitter, TensorRecorder
        recorder = TensorRecorder(
            args.record_to,
            source="futudiffu",
            config_metadata={
                "prompt": config.prompt,
                "seed": config.seed,
                "steps": config.steps,
                "cfg": config.cfg,
                "width": config.width,
                "height": config.height,
                "fp8_diffusion": config.fp8_diffusion,
                "fp8_text_encoder": config.fp8_text_encoder,
                "comfyui_compat": config.comfyui_compat,
            },
        )
        emitter = TensorEmitter([recorder])
        print(f"Recording tensor stream to {args.record_to}")

    image_np = generate(config, emitter=emitter)

    if emitter is not None:
        emitter.close()
        print(f"Tensor stream saved to {args.record_to}")

    # Save as PNG
    from PIL import Image
    img = Image.fromarray(image_np[0])
    img.save(args.output)
    print(f"Saved to {args.output}")

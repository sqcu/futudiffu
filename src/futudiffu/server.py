"""Inference server: long-running process that owns GPU and runs inference.

ZeroMQ REP socket on localhost. Accepts requests via the protocol module,
dispatches to the appropriate model, returns results as serialized tensors.

Manages model lifecycle: loads/frees models as needed to fit in VRAM.
Model groups are mutually exclusive for large allocations:
  - "te" phase: text encoder (~7.5GB)
  - "diffusion" phase: FP8 diffusion model (~8GB with compile) + optionally VAE
  - "vae" phase: VAE only (~320MB)

Usage:
    python -m futudiffu.server --port 5555 \\
        --fp8-diff /path/to/z_image_fp8_blockwise.safetensors \\
        --te /path/to/qwen_3_4b.safetensors \\
        --vae /path/to/zimage.safetensors
"""

import argparse
import sys
import time
import traceback

import torch
import torch.nn.functional as F
import zmq

from .protocol import pack_response, unpack_request


class InferenceServer:
    """GPU-owning inference server with model lifecycle management."""

    def __init__(
        self,
        fp8_diff_path: str,
        te_path: str,
        vae_path: str,
        tokenizer_path: str | None = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
        fp8_block_size: int = 128,
    ):
        self.fp8_diff_path = fp8_diff_path
        self.te_path = te_path
        self.vae_path = vae_path
        self.tokenizer_path = tokenizer_path
        self.device = torch.device(device)
        self.dtype = {"float32": torch.float32, "float16": torch.float16,
                      "bfloat16": torch.bfloat16}[dtype]
        self.fp8_block_size = fp8_block_size

        # Model state -- at most one large model loaded at a time
        self._te_model = None
        self._tokenizer = None
        self._diff_model = None        # raw model
        self._diff_compiled = None      # torch.compile'd forward()
        self._diff_compiled_packed = None  # torch.compile'd forward_packed()
        self._vae_model = None

        # Which phase we're in determines what's loaded
        self._phase = None  # "te" | "diffusion" | "vae" | None

        # Sage configured once at startup
        self._sage_configured = False

        # LoRA lifecycle: survive model swaps (TE <-> diffusion)
        self._lora_configs: list[dict] = []  # Injection configs to replay on reload
        self._lora_weights: dict[str, dict[str, torch.Tensor]] = {}  # CPU weights per adapter
        self._lora_scales: dict[str, float | list[float]] = {}  # Last-set scale per adapter

        # BTRM head: lives alongside LoRA, persisted together (~30KB, permanent GPU resident)
        self._btrm_head: torch.nn.Module | None = None
        self._btrm_optimizer: torch.optim.Optimizer | None = None
        self._btrm_config: dict | None = None  # For persistence/crash dump

        # Policy optimizer: lazy-init on first policy_optimizer_step call
        self._policy_optimizers: dict[str, torch.optim.Optimizer] = {}

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _snapshot_lora_weights(self):
        """Save current LoRA weights to CPU before destroying diffusion model."""
        if self._diff_model is not None and self._lora_configs:
            from .lora import lora_state_dict
            for cfg in self._lora_configs:
                name = cfg["adapter_name"]
                sd = lora_state_dict(self._diff_model, adapter_name=name)
                self._lora_weights[name] = {
                    k: v.detach().cpu() for k, v in sd.items()
                }

    def _free_all(self):
        """Free all models from VRAM."""
        if self._te_model is not None:
            del self._te_model
            self._te_model = None
            self._tokenizer = None
        if self._diff_model is not None:
            self._snapshot_lora_weights()
            del self._diff_model, self._diff_compiled, self._diff_compiled_packed
            self._diff_model = None
            self._diff_compiled = None
            self._diff_compiled_packed = None
        if self._vae_model is not None:
            del self._vae_model
            self._vae_model = None
        self._phase = None
        torch.cuda.empty_cache()

    def _ensure_te(self):
        """Ensure text encoder is loaded. Frees other large models if needed."""
        if self._te_model is not None:
            return
        # Free diffusion (large) but keep VAE (small, can coexist)
        if self._diff_model is not None:
            self._snapshot_lora_weights()
            del self._diff_model, self._diff_compiled, self._diff_compiled_packed
            self._diff_model = None
            self._diff_compiled = None
            self._diff_compiled_packed = None
            torch.cuda.empty_cache()

        from .text_encoder import create_tokenizer, load_text_encoder

        self._tokenizer = create_tokenizer(self.tokenizer_path)
        self._te_model = load_text_encoder(
            self.te_path, device=self.device, dtype=self.dtype
        )
        self._te_model = torch.compile(self._te_model, mode="default")
        self._phase = "te"
        print(f"  [lifecycle] TE loaded ({self.dtype})")

    def _free_te(self):
        """Free text encoder."""
        if self._te_model is not None:
            del self._te_model
            self._te_model = None
            self._tokenizer = None
            torch.cuda.empty_cache()

    def _ensure_diffusion(self):
        """Ensure diffusion model is loaded. Frees TE if needed."""
        if self._diff_model is not None:
            return
        # Free TE (large)
        self._free_te()

        from .diffusion_model import (
            _detect_cap_feat_dim,
            _detect_n_layers,
            _detect_qk_norm,
            _strip_diffusion_prefix,
            create_diffusion_model,
            fuse_model,
        )
        from .fp8 import replace_linear_with_fp8
        from safetensors.torch import load_file

        t0 = time.perf_counter()
        diff_sd = load_file(self.fp8_diff_path, device=str(self.device))
        remapped = _strip_diffusion_prefix(diff_sd)
        del diff_sd

        n_layers = _detect_n_layers(remapped.keys())
        cap_feat_dim = _detect_cap_feat_dim(remapped)
        qk_norm = _detect_qk_norm(remapped.keys())
        model = create_diffusion_model(
            dtype=self.dtype, n_layers=n_layers,
            cap_feat_dim=cap_feat_dim, qk_norm=qk_norm,
        )
        replace_linear_with_fp8(
            model, remapped, block_size=self.fp8_block_size,
            output_dtype=self.dtype,
        )

        remaining = {k: v for k, v in remapped.items()
                     if not k.endswith((".weight_scale", ".comfy_quant"))}
        model.load_state_dict(remaining, strict=False, assign=True)
        del remapped, remaining

        model = model.to(self.device)
        model.eval()
        fuse_model(model)
        self._diff_model = model

        # Replay LoRA injections that were active before the lifecycle swap.
        # fuse_model() must run first because LoRA targets post-fusion modules
        # (e.g. the fused w1w3 FP8Linear).
        if self._lora_configs:
            from .lora import inject_lora, load_lora_state_dict, set_lora_scale
            for cfg in self._lora_configs:
                li = cfg["layer_indices"]
                inject_lora(
                    model,
                    name=cfg["adapter_name"],
                    rank=cfg["rank"],
                    alpha=cfg["alpha"],
                    layer_indices=li,
                    init_b_std=0.0,  # Don't re-randomize; we load saved weights
                )
            # Restore saved weights
            for adapter_name, weights in self._lora_weights.items():
                if weights:
                    # Move weights to model dtype for load
                    sd = {k: v.to(dtype=self.dtype) for k, v in weights.items()}
                    load_lora_state_dict(model, sd)
            # Restore scale settings
            for adapter_name, scale in self._lora_scales.items():
                if isinstance(scale, (int, float)):
                    scale_tensor = torch.tensor(
                        [scale], device=self.device, dtype=self.dtype)
                else:
                    scale_tensor = torch.tensor(
                        scale, device=self.device, dtype=self.dtype)
                set_lora_scale(model, scale_tensor, adapter_name=adapter_name)
            print(f"  [lifecycle] Replayed {len(self._lora_configs)} LoRA injection(s)")

        # torch.compile after LoRA replay so the compiled graph includes
        # LoRA modules from the start, avoiding an extra dynamo reset.
        self._diff_compiled = torch.compile(model, mode="default")
        # forward_packed is a separate method — torch.compile(model) only
        # wraps forward(). Compile it separately for FlexAttention perf.
        self._diff_compiled_packed = torch.compile(
            model.forward_packed, mode="default"
        )
        self._phase = "diffusion"
        elapsed = time.perf_counter() - t0
        print(f"  [lifecycle] Diffusion loaded (FP8, fused, compiled) in {elapsed:.1f}s")

    def _ensure_vae(self):
        """Ensure VAE is loaded. VAE is small enough to coexist with diffusion."""
        if self._vae_model is not None:
            return
        from .vae import load_vae

        self._vae_model = load_vae(
            self.vae_path, device=self.device, dtype=self.dtype
        )
        print(f"  [lifecycle] VAE loaded")

    # ------------------------------------------------------------------
    # RPC methods
    # ------------------------------------------------------------------

    def handle_encode_prompt(self, params, tensors):
        """Encode a text prompt to conditioning tensor.

        Params:
            prompt (str): Text to encode.
            layer_idx (int, optional): Hidden layer index, default -2.

        Returns:
            Tensor "conditioning": (1, seq_len, 2560)
        """
        self._ensure_te()
        from .text_encoder import encode_prompt

        prompt = params["prompt"]
        layer_idx = params.get("layer_idx", -2)

        with torch.inference_mode():
            cond = encode_prompt(
                self._te_model, self._tokenizer, prompt,
                device=self.device, layer_idx=layer_idx,
            )
        return pack_response("ok", {"conditioning": cond})

    def handle_sample_trajectory(self, params, tensors):
        """Run a diffusion sampling trajectory.

        Params:
            seed (int): RNG seed for noise generation.
            n_steps (int): Number of euler steps.
            cfg (float): CFG scale.
            width (int): Image width.
            height (int): Image height.
            attention_backend (str): "sdpa" or "sage".
            sampling_shift (float, optional): Default 1.0.
            multiplier (float, optional): Default 1.0.
            save_steps (list[int] | null): Steps to save intermediates. null = all.
            denoise (float, optional): For i2i, denoise strength (0-1). Default 1.0.

        Tensors:
            pos_cond: (1, seq, 2560) positive conditioning.
            neg_cond: (1, seq, 2560) negative conditioning.
            clean_latent: (1, 16, H/8, W/8) optional, for i2i.

        Returns:
            Tensors: "final" + "step_NN" for each saved step.
            Metadata: {"saved_steps": [...]}
        """
        from .attention import set_attention_backend
        from .sampling import (
            build_sigmas,
            const_calculate_denoised,
            const_inverse_noise_scaling,
            const_noise_scaling,
            sample_euler,
            simple_scheduler,
        )

        self._ensure_diffusion()

        seed = params["seed"]
        n_steps = params["n_steps"]
        cfg = params["cfg"]
        width = params["width"]
        height = params["height"]
        attn_backend = params.get("attention_backend", "sdpa")
        sampling_shift = params.get("sampling_shift", 1.0)
        multiplier = params.get("multiplier", 1.0)
        save_steps = params.get("save_steps", None)
        denoise = params.get("denoise", 1.0)

        pos_cond = tensors["pos_cond"].to(device=self.device, dtype=self.dtype)
        neg_cond = tensors["neg_cond"].to(device=self.device, dtype=self.dtype)
        clean_latent = tensors.get("clean_latent")
        if clean_latent is not None:
            clean_latent = clean_latent.to(device=self.device, dtype=self.dtype)

        # Configure attention backend
        if not self._sage_configured and attn_backend != "sdpa":
            from .sage_attention import configure_sage
            configure_sage(smooth_k=True, qk_quant="int8", pv_quant="bf16")
            self._sage_configured = True
        set_attention_backend(attn_backend)

        device = self.device
        dtype = self.dtype
        # lora_scale is a registered buffer — torch.compile tracks it
        # as a dynamic graph input.  No need for use_raw_model bypass.
        diff_model = self._diff_compiled

        latent_h = height // 8
        latent_w = width // 8

        # Pad conditioning to same length for batched CFG
        pos_len = pos_cond.shape[1]
        neg_len = neg_cond.shape[1]
        max_len = max(pos_len, neg_len)
        if pos_len < max_len:
            pos_cond = F.pad(pos_cond, (0, 0, 0, max_len - pos_len))
        if neg_len < max_len:
            neg_cond = F.pad(neg_cond, (0, 0, 0, max_len - neg_len))
        num_tokens = max_len
        cond_batch = torch.cat([pos_cond, neg_cond], dim=0)

        # RoPE cache
        padded_h = latent_h + ((-latent_h) % self._diff_model.patch_size)
        padded_w = latent_w + ((-latent_w) % self._diff_model.patch_size)
        rope_cache = self._diff_model.prepare_rope_cache(
            padded_h, padded_w, num_tokens, device
        )

        # Noise
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(
            1, 16, latent_h, latent_w, dtype=dtype,
            generator=generator, device=device,
        )

        # Sigma schedule
        sigma_table = build_sigmas(shift=sampling_shift, multiplier=multiplier * 1000)

        if denoise < 1.0:
            # i2i: ComfyUI-style expanded schedule.
            # Build a longer schedule, take the last (n_steps + 1) sigmas.
            # This starts denoising from sigma ≈ denoise, running all n_steps
            # iterations with appropriately-spaced sigmas.
            expanded_steps = int(n_steps / denoise)
            full_sigmas = simple_scheduler(sigma_table, expanded_steps)
            full_sigmas = full_sigmas.to(device=device, dtype=dtype)
            sigmas = full_sigmas[-(n_steps + 1):]
        else:
            sigmas = simple_scheduler(sigma_table, n_steps)
            sigmas = sigmas.to(device=device, dtype=dtype)

        # Initial latent
        if clean_latent is not None:
            x = const_noise_scaling(sigmas[0], noise, clean_latent)
        else:
            latent = torch.zeros(1, 16, latent_h, latent_w, device=device, dtype=dtype)
            x = const_noise_scaling(sigmas[0], noise, latent)

        # CFG model function
        def model_fn(x_in, sigma):
            timestep = sigma * multiplier
            x_batch = x_in.expand(2, -1, -1, -1)
            t_batch = timestep.expand(2)
            output_batch = diff_model(
                x_batch, t_batch, cond_batch,
                num_tokens=num_tokens, rope_cache=rope_cache,
            )
            out_cond, out_uncond = output_batch.chunk(2, dim=0)
            denoised_cond = const_calculate_denoised(sigma, out_cond, x_in)
            denoised_uncond = const_calculate_denoised(sigma, out_uncond, x_in)
            return denoised_uncond + (denoised_cond - denoised_uncond) * cfg

        # Determine which steps to save
        if save_steps is not None:
            steps_to_save = {s for s in save_steps if s < n_steps}
        else:
            steps_to_save = set(range(n_steps))

        result_tensors = {}

        def save_callback(info):
            i = info["i"]
            if i in steps_to_save:
                result_tensors[f"step_{i:02d}"] = info["x"].detach().cpu()

        with torch.inference_mode():
            x = sample_euler(model_fn, x, sigmas, callback=save_callback)
            x = const_inverse_noise_scaling(sigmas[-1], x)

        result_tensors["final"] = x.detach().cpu()

        saved = sorted(k for k in result_tensors if k.startswith("step_"))
        return pack_response(
            "ok", result_tensors, {"saved_steps": saved}
        )

    def handle_sample_trajectory_packed(self, params, tensors):
        """Run N diffusion trajectories packed via FlexAttention.

        All trajectories share the same sigma schedule (n_steps, denoise,
        sampling_shift, multiplier, cfg).  Each has its own prompt and seed.

        Params:
            n_images (int): Number of images to pack.
            seeds (list[int]): Per-image RNG seeds.
            n_steps (int): Euler steps (shared).
            cfg (float): CFG scale (shared).
            width (int): Image width (shared, all same size for now).
            height (int): Image height (shared).
            sampling_shift (float, optional): Default 1.0.
            multiplier (float, optional): Default 1.0.
            save_steps (list[int] | null): Steps to save intermediates.
            denoise (float, optional): For i2i, denoise strength. Default 1.0.

        Tensors:
            pos_cond_0 .. pos_cond_{N-1}: Per-image positive conditioning.
            neg_cond: Shared negative conditioning (1, seq, 2560).
            clean_latent_0 .. clean_latent_{N-1}: Optional, for i2i.

        Returns:
            Tensors: "final_0" .. "final_{N-1}", plus "step_NN_I" per saved step.
            Metadata: {"n_images": N, "saved_steps": [...]}
        """
        from .diffusion_model import make_packing_mask_mod, pad_to_patch_size
        from .sampling import (
            build_sigmas,
            const_calculate_denoised,
            const_inverse_noise_scaling,
            const_noise_scaling,
            simple_scheduler,
            to_d,
        )
        from torch.nn.attention.flex_attention import create_block_mask

        self._ensure_diffusion()

        n_images = params["n_images"]
        seeds = params["seeds"]
        n_steps = params["n_steps"]
        cfg = params["cfg"]
        width = params["width"]
        height = params["height"]
        sampling_shift = params.get("sampling_shift", 1.0)
        multiplier = params.get("multiplier", 1.0)
        save_steps_param = params.get("save_steps", None)
        denoise = params.get("denoise", 1.0)

        device = self.device
        dtype = self.dtype
        pH = pW = self._diff_model.patch_size

        # Collect per-image conditioning
        neg_cond = tensors["neg_cond"].to(device=device, dtype=dtype)
        neg_len = neg_cond.shape[1]

        # Build CFG conditionings: (2, max_len_i, dim) per image
        # batch[0] = pos, batch[1] = neg
        cfg_conds = []
        cap_lens = []
        for i in range(n_images):
            pos_i = tensors[f"pos_cond_{i}"].to(device=device, dtype=dtype)
            pos_len = pos_i.shape[1]
            max_len = max(pos_len, neg_len)
            if pos_len < max_len:
                pos_i = F.pad(pos_i, (0, 0, 0, max_len - pos_len))
            neg_padded = neg_cond
            if neg_len < max_len:
                neg_padded = F.pad(neg_cond, (0, 0, 0, max_len - neg_len))
            cfg_conds.append(torch.cat([pos_i, neg_padded], dim=0))  # (2, max_len, dim)
            cap_lens.append(max_len)

        # Latent sizes
        latent_h = height // 8
        latent_w = width // 8
        padded_h = latent_h + ((-latent_h) % pH)
        padded_w = latent_w + ((-latent_w) % pW)

        # Noise + initial latent per image
        x_list = []
        for i in range(n_images):
            gen = torch.Generator(device=device).manual_seed(seeds[i])
            noise = torch.randn(
                1, 16, latent_h, latent_w, dtype=dtype,
                generator=gen, device=device,
            )
            clean = tensors.get(f"clean_latent_{i}")
            if clean is not None:
                clean = clean.to(device=device, dtype=dtype)
            else:
                clean = torch.zeros(1, 16, latent_h, latent_w, device=device, dtype=dtype)
            x_list.append(noise)  # stored for noise_scaling below
            del clean  # reused below with sigmas

        # Sigma schedule
        sigma_table = build_sigmas(shift=sampling_shift, multiplier=multiplier * 1000)
        if denoise < 1.0:
            expanded_steps = int(n_steps / denoise)
            full_sigmas = simple_scheduler(sigma_table, expanded_steps)
            full_sigmas = full_sigmas.to(device=device, dtype=dtype)
            sigmas = full_sigmas[-(n_steps + 1):]
        else:
            sigmas = simple_scheduler(sigma_table, n_steps)
            sigmas = sigmas.to(device=device, dtype=dtype)

        # Initial noise scaling
        for i in range(n_images):
            noise_i = x_list[i]
            clean_i = tensors.get(f"clean_latent_{i}")
            if clean_i is not None:
                clean_i = clean_i.to(device=device, dtype=dtype)
            else:
                clean_i = torch.zeros_like(noise_i)
            x_list[i] = const_noise_scaling(sigmas[0], noise_i, clean_i)

        # Prepare packing state (constant across euler steps)
        with torch.inference_mode():
            padded_sizes = [(padded_h, padded_w)] * n_images
            refined_caps, packing_info, packed_rope = \
                self._diff_model.prepare_packed_state(
                    cfg_conds, padded_sizes, cap_lens, device,
                )

            block_mask = create_block_mask(
                make_packing_mask_mod(packing_info.document_id),
                B=2, H=None,
                Q_LEN=packing_info.total_len,
                KV_LEN=packing_info.total_len,
                device=device,
            )

        # Determine which steps to save
        if save_steps_param is not None:
            steps_to_save = {s for s in save_steps_param if s < n_steps}
        else:
            steps_to_save = set(range(n_steps))

        result_tensors = {}

        # Euler loop over packed forward passes
        with torch.inference_mode():
            for step_i in range(n_steps):
                sigma = sigmas[step_i]
                timestep = sigma * multiplier

                # Expand x for CFG: each (1, C, H, W) -> (2, C, H, W)
                x_cfg = [x_i.expand(2, -1, -1, -1) for x_i in x_list]
                t_batch = timestep.expand(2)

                # Packed forward pass
                outputs = self._diff_compiled_packed(
                    x_cfg, t_batch, refined_caps,
                    packing_info, block_mask, packed_rope,
                )

                # CFG per image + euler step
                for img_i in range(n_images):
                    out_cond, out_uncond = outputs[img_i].chunk(2, dim=0)
                    denoised_cond = const_calculate_denoised(
                        sigma, out_cond, x_list[img_i])
                    denoised_uncond = const_calculate_denoised(
                        sigma, out_uncond, x_list[img_i])
                    denoised = denoised_uncond + (denoised_cond - denoised_uncond) * cfg

                    d = to_d(x_list[img_i], sigma, denoised)
                    dt = sigmas[step_i + 1] - sigma
                    x_list[img_i] = x_list[img_i] + d * dt

                # Save step checkpoints
                if step_i in steps_to_save:
                    for img_i in range(n_images):
                        result_tensors[f"step_{step_i:02d}_{img_i}"] = \
                            x_list[img_i].detach().cpu()

            # Inverse noise scaling
            for img_i in range(n_images):
                x_list[img_i] = const_inverse_noise_scaling(
                    sigmas[-1], x_list[img_i])
                result_tensors[f"final_{img_i}"] = x_list[img_i].detach().cpu()

        saved = sorted(k for k in result_tensors if k.startswith("step_"))
        return pack_response(
            "ok", result_tensors,
            {"n_images": n_images, "saved_steps": saved},
        )

    def handle_vae_encode(self, params, tensors):
        """Encode image to latent.

        Tensors:
            image: (1, 3, H, W) in [0, 1] range.

        Returns:
            Tensor "latent": (1, 16, H/8, W/8).
        """
        self._ensure_vae()
        from .vae import vae_encode

        image = tensors["image"].to(device=self.device, dtype=self.dtype)
        with torch.inference_mode():
            latent = vae_encode(self._vae_model, image)
        return pack_response("ok", {"latent": latent.cpu()})

    def handle_vae_decode(self, params, tensors):
        """Decode latent to image.

        Tensors:
            latent: (1, 16, H, W).

        Returns:
            Tensor "image": (1, 3, H*8, W*8) in [0, 1] range.
        """
        self._ensure_vae()
        from .vae import vae_decode

        latent = tensors["latent"].to(device=self.device, dtype=self.dtype)
        with torch.inference_mode():
            image = vae_decode(self._vae_model, latent)
        return pack_response("ok", {"image": image.cpu()})

    def handle_warmup(self, params, tensors):
        """Warmup the diffusion model with a short trajectory.

        Params:
            attention_backend (str): "sdpa" or "sage".
        """
        from .attention import set_attention_backend
        from .sampling import (
            build_sigmas,
            const_calculate_denoised,
            const_noise_scaling,
            sample_euler,
            simple_scheduler,
        )

        self._ensure_diffusion()

        attn_backend = params.get("attention_backend", "sdpa")
        if not self._sage_configured and attn_backend != "sdpa":
            from .sage_attention import configure_sage
            configure_sage(smooth_k=True, qk_quant="int8", pv_quant="bf16")
            self._sage_configured = True
        set_attention_backend(attn_backend)

        device = self.device
        dtype = self.dtype
        diff_model = self._diff_compiled

        # Use small fixed params for warmup
        latent_h, latent_w = 104, 160  # 832x1280
        width, height = 1280, 832

        # Dummy conditioning (short sequence, correct dim)
        dummy_cond = torch.zeros(2, 32, 2560, device=device, dtype=dtype)
        num_tokens = 32

        padded_h = latent_h + ((-latent_h) % self._diff_model.patch_size)
        padded_w = latent_w + ((-latent_w) % self._diff_model.patch_size)
        rope_cache = self._diff_model.prepare_rope_cache(
            padded_h, padded_w, num_tokens, device
        )

        noise = torch.randn(1, 16, latent_h, latent_w, dtype=dtype, device=device)
        sigma_table = build_sigmas(shift=1.0, multiplier=1000.0)
        sigmas = simple_scheduler(sigma_table, 4).to(device=device, dtype=dtype)
        latent = torch.zeros(1, 16, latent_h, latent_w, device=device, dtype=dtype)
        x = const_noise_scaling(sigmas[0], noise, latent)

        def model_fn(x_in, sigma):
            timestep = sigma * 1.0
            x_batch = x_in.expand(2, -1, -1, -1)
            t_batch = timestep.expand(2)
            output_batch = diff_model(
                x_batch, t_batch, dummy_cond,
                num_tokens=num_tokens, rope_cache=rope_cache,
            )
            out_cond, out_uncond = output_batch.chunk(2, dim=0)
            denoised_cond = const_calculate_denoised(sigma, out_cond, x_in)
            denoised_uncond = const_calculate_denoised(sigma, out_uncond, x_in)
            return denoised_uncond + (denoised_cond - denoised_uncond) * 4.0

        with torch.inference_mode():
            sample_euler(model_fn, x, sigmas)
        torch.cuda.synchronize()
        print(f"  [warmup] {attn_backend} done")
        return pack_response("ok")

    def handle_warmup_packed(self, params, tensors):
        """Warmup the packed forward path (FlexAttention + torch.compile).

        Runs a single packed forward pass with 2 small dummy images to
        trigger compilation of forward_packed via FlexAttention.

        Params:
            n_images (int, optional): Number of dummy images (default 2).
        """
        from .diffusion_model import make_packing_mask_mod
        from .sampling import const_noise_scaling
        from torch.nn.attention.flex_attention import create_block_mask

        self._ensure_diffusion()

        device = self.device
        dtype = self.dtype
        n = params.get("n_images", 2)
        pH = pW = self._diff_model.patch_size

        # Small dummy images (256x256 latent = 32x32)
        latent_h, latent_w = 32, 32
        padded_h = latent_h + ((-latent_h) % pH)
        padded_w = latent_w + ((-latent_w) % pW)

        # Dummy CFG conditionings: (2, 32, 2560) per image
        cfg_conds = [
            torch.zeros(2, 32, 2560, device=device, dtype=dtype)
            for _ in range(n)
        ]
        cap_lens = [32] * n

        with torch.inference_mode():
            refined_caps, packing_info, packed_rope = \
                self._diff_model.prepare_packed_state(
                    cfg_conds, [(padded_h, padded_w)] * n, cap_lens, device,
                )

            block_mask = create_block_mask(
                make_packing_mask_mod(packing_info.document_id),
                B=2, H=None,
                Q_LEN=packing_info.total_len,
                KV_LEN=packing_info.total_len,
                device=device,
            )

            x_list = [
                torch.randn(2, 16, latent_h, latent_w, device=device, dtype=dtype)
                for _ in range(n)
            ]
            t_batch = torch.tensor([0.5, 0.5], device=device, dtype=dtype)

            t0 = time.perf_counter()
            self._diff_compiled_packed(
                x_list, t_batch, refined_caps,
                packing_info, block_mask, packed_rope,
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

        print(f"  [warmup_packed] n={n}, {elapsed:.1f}s (includes compilation)")
        return pack_response("ok")

    def handle_status(self, params, tensors):
        """Return server status: loaded models, VRAM stats."""
        vram_allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
        vram_reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
        vram_total = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)

        loaded = []
        if self._te_model is not None:
            loaded.append("te")
        if self._diff_model is not None:
            loaded.append("diffusion")
        if self._vae_model is not None:
            loaded.append("vae")

        info = {
            "loaded_models": loaded,
            "phase": self._phase,
            "vram_allocated_gb": round(vram_allocated, 2),
            "vram_reserved_gb": round(vram_reserved, 2),
            "vram_total_gb": round(vram_total, 2),
            "sage_configured": self._sage_configured,
        }
        return pack_response("ok", metadata=info)

    # ------------------------------------------------------------------
    # LoRA management RPCs
    # ------------------------------------------------------------------

    def handle_inject_lora(self, params, tensors):
        """Inject a named LoRA adapter into the diffusion model.

        Params:
            adapter_name (str): Name for the adapter (e.g. "ptheta", "rtheta").
            rank (int): LoRA rank.
            alpha (float): LoRA alpha.
            layer_indices (list[int] | null): If set, only inject on these layers.

        After injection, torch.compile caches are invalidated. Client must
        call warmup again.
        """
        from .lora import inject_lora

        self._ensure_diffusion()

        adapter_name = params["adapter_name"]
        rank = params.get("rank", 8)
        alpha = params.get("alpha", 16.0)
        layer_indices = params.get("layer_indices")
        init_b_std = params.get("init_b_std", 0.0)
        if layer_indices is not None:
            layer_indices = set(layer_indices)

        injected = inject_lora(
            self._diff_model,
            name=adapter_name,
            rank=rank,
            alpha=alpha,
            layer_indices=layer_indices,
            init_b_std=init_b_std,
        )

        # Record config for lifecycle replay
        self._lora_configs.append({
            "adapter_name": adapter_name,
            "rank": rank,
            "alpha": alpha,
            "layer_indices": layer_indices,  # Already a set or None
            "init_b_std": init_b_std,
        })

        # Module tree changed -- invalidate compiled versions
        torch._dynamo.reset()
        self._diff_compiled = torch.compile(self._diff_model, mode="default")
        self._diff_compiled_packed = torch.compile(
            self._diff_model.forward_packed, mode="default"
        )

        n_params = sum(
            a.lora_A.numel() + a.lora_B.numel() for a in injected.values()
        )
        print(f"  [inject_lora] {adapter_name}: {len(injected)} adapters, "
              f"{n_params:,} params")

        return pack_response("ok", metadata={
            "adapter_name": adapter_name,
            "n_adapters": len(injected),
            "n_params": n_params,
        })

    def handle_update_lora_weights(self, params, tensors):
        """Update LoRA weights in-place (hot path during training).

        Uses .data.copy_() so tensor identity is preserved. NO dynamo reset
        needed -- CUDA graphs remain valid.

        Tensors:
            Keys matching "path.adapters.name.lora_A" / "lora_B" format.
        """
        from .lora import load_lora_state_dict

        self._ensure_diffusion()

        # Convert tensors to the model's dtype and load
        sd = {k: v.to(dtype=self.dtype) for k, v in tensors.items()}
        load_lora_state_dict(self._diff_model, sd)

        # Update the CPU weight cache so lifecycle reload gets latest weights.
        # Keys are "path.adapters.<name>.lora_A" / "lora_B".
        for key, val in sd.items():
            parts = key.split(".adapters.")
            if len(parts) == 2:
                adapter_name = parts[1].split(".")[0]
                if adapter_name not in self._lora_weights:
                    self._lora_weights[adapter_name] = {}
                self._lora_weights[adapter_name][key] = val.detach().cpu()

        print(f"  [update_lora_weights] {len(sd)} tensors updated")
        return pack_response("ok", metadata={"n_tensors": len(sd)})

    def handle_set_adapter_config(self, params, tensors):
        """Set adapter scale and/or freeze state.

        No dynamo reset needed — lora_scale is a registered buffer, which
        torch.compile treats as a graph input operand.  Same-shape value
        changes are free (in-place copy_).  Shape changes (e.g. scalar to
        per-batch) trigger one recompilation.

        Params:
            adapter_name (str): Which adapter to configure.
            scale (float | list[float] | null): Per-batch scale. null = clear.
            frozen (bool | null): If set, freeze or unfreeze the adapter.
        """
        from .lora import set_lora_scale, clear_lora_scale, freeze_adapter

        self._ensure_diffusion()

        adapter_name = params["adapter_name"]
        scale = params.get("scale")
        frozen = params.get("frozen")

        if scale is not None:
            if isinstance(scale, (int, float)):
                scale_tensor = torch.tensor([scale], device=self.device, dtype=self.dtype)
            else:
                scale_tensor = torch.tensor(scale, device=self.device, dtype=self.dtype)
            set_lora_scale(self._diff_model, scale_tensor, adapter_name=adapter_name)
            # Persist scale for lifecycle replay
            self._lora_scales[adapter_name] = scale
        elif scale is None and "scale" in params:
            clear_lora_scale(self._diff_model, adapter_name=adapter_name)
            # Clear persisted scale
            self._lora_scales.pop(adapter_name, None)

        n_frozen = 0
        if frozen is not None and frozen:
            n_frozen = freeze_adapter(self._diff_model, adapter_name)

        print(f"  [set_adapter_config] {adapter_name}: scale={scale}, "
              f"frozen={frozen} ({n_frozen} adapters)")
        return pack_response("ok", metadata={
            "adapter_name": adapter_name,
            "n_frozen": n_frozen,
        })

    # ------------------------------------------------------------------
    # Training-support RPCs
    # ------------------------------------------------------------------

    def _run_backbone_hidden(self, latent, sigma, conditioning, multiplier=1.0,
                             attention_backend="sdpa"):
        """Run frozen backbone and return hidden states from last transformer block.

        Shared implementation for score_btrm, train_btrm_step, etc.

        Returns:
            Hidden states (B, N_tokens, hidden_dim) on GPU.
        """
        from .attention import set_attention_backend

        if not self._sage_configured and attention_backend != "sdpa":
            from .sage_attention import configure_sage
            configure_sage(smooth_k=True, qk_quant="int8", pv_quant="bf16")
            self._sage_configured = True
        set_attention_backend(attention_backend)

        latent = latent.to(device=self.device, dtype=self.dtype)
        sigma = sigma.to(device=self.device, dtype=self.dtype)
        conditioning = conditioning.to(device=self.device, dtype=self.dtype)

        timestep = sigma * multiplier
        num_tokens = conditioning.shape[1]

        B, C, H, W = latent.shape
        padded_h = H + ((-H) % self._diff_model.patch_size)
        padded_w = W + ((-W) % self._diff_model.patch_size)
        rope_cache = self._diff_model.prepare_rope_cache(
            padded_h, padded_w, num_tokens, self.device
        )

        captured = [None]
        last_block = self._diff_model.layers[-1]

        def hook(_module, _input, output):
            captured[0] = output

        handle = last_block.register_forward_hook(hook)
        try:
            with torch.inference_mode():
                self._diff_model(
                    latent, timestep, conditioning,
                    num_tokens=num_tokens, rope_cache=rope_cache,
                )
        finally:
            handle.remove()

        hidden = captured[0]
        assert hidden is not None, "Hook did not fire on layers[-1]"
        return hidden

    def handle_inject_btrm_head(self, params, tensors):
        """Create a BTRM scoring head on the server.

        The head is ~30KB and stays on GPU permanently alongside LoRA.

        Params:
            hidden_dim (int, optional): Default 3840.
            head_names (list[str], optional): Default ["bit_quality","step_quality"].
            logit_cap (float, optional): Default 10.0.
            lr (float, optional): If set, create an Adam optimizer for BTRM training.
            weight_decay (float, optional): Default 0.

        Returns:
            Metadata: {n_heads, n_params}
        """
        from .btrm import BTRMHead

        hidden_dim = params.get("hidden_dim", 3840)
        head_names = params.get("head_names", ["bit_quality", "step_quality"])
        logit_cap = params.get("logit_cap", 10.0)
        lr = params.get("lr")
        weight_decay = params.get("weight_decay", 0.0)

        self._btrm_head = BTRMHead(
            hidden_dim=hidden_dim,
            head_names=head_names,
            logit_cap=logit_cap,
        ).to(device=self.device, dtype=self.dtype)
        self._btrm_head.train()

        self._btrm_config = {
            "hidden_dim": hidden_dim,
            "head_names": list(head_names),
            "logit_cap": logit_cap,
        }

        if lr is not None:
            self._btrm_optimizer = torch.optim.AdamW(
                self._btrm_head.parameters(), lr=lr, weight_decay=weight_decay,
            )

        n_params = sum(p.numel() for p in self._btrm_head.parameters())
        print(f"  [inject_btrm_head] {len(head_names)} heads, {n_params:,} params"
              f"{f', optimizer lr={lr}' if lr else ''}")

        return pack_response("ok", metadata={
            "n_heads": len(head_names),
            "n_params": n_params,
        })

    def handle_score_btrm(self, params, tensors):
        """Run backbone + BTRM head, return scores as metadata (no tensor output).

        Params:
            attention_backend (str, optional): Default "sdpa".
            multiplier (float, optional): Default 1.0.

        Tensors:
            latent: (B, 16, H, W) noisy latent.
            sigma: (B,) sigma values.
            conditioning: (B, seq, dim) text conditioning.

        Returns:
            Metadata: {scores: [[head0, head1], ...]}
        """
        self._ensure_diffusion()
        assert self._btrm_head is not None, "BTRM head not injected. Call inject_btrm_head first."

        multiplier = params.get("multiplier", 1.0)
        attn_backend = params.get("attention_backend", "sdpa")

        hidden = self._run_backbone_hidden(
            tensors["latent"], tensors["sigma"], tensors["conditioning"],
            multiplier=multiplier, attention_backend=attn_backend,
        )

        with torch.no_grad():
            scores = self._btrm_head(hidden)  # (B, N_heads)

        scores_list = scores.detach().cpu().tolist()
        return pack_response("ok", metadata={"scores": scores_list})

    def handle_train_btrm_step(self, params, tensors):
        """One BTRM optimizer step from labeled examples.

        Server does everything: forward through backbone, score, compute
        all-pairs BT loss per head, backward, optimizer step.

        Params:
            labels (list[dict]): Per-example labels. Each dict has:
                head_idx (int): Which head this example trains.
                is_positive (bool): Whether this is a positive example.
            logsquare_weight (float, optional): Default 0.1.
            attention_backend (str, optional): Default "sdpa".
            multiplier (float, optional): Default 1.0.

        Tensors:
            latent_0..N: (1, 16, H, W) per-example noisy latent.
            sigma_0..N: (1,) per-example sigma.
            conditioning_0..N: (1, seq, dim) per-example conditioning.

        Returns:
            Metadata: {loss, bt_loss, logsq_loss, per_head_accuracy, n_examples}
        """
        from .btrm import bt_loss_allpairs, logsquare_regularizer

        self._ensure_diffusion()
        assert self._btrm_head is not None, "BTRM head not injected"
        assert self._btrm_optimizer is not None, "BTRM optimizer not created (pass lr to inject_btrm_head)"

        labels = params["labels"]
        logsq_weight = params.get("logsquare_weight", 0.1)
        attn_backend = params.get("attention_backend", "sdpa")
        multiplier = params.get("multiplier", 1.0)
        n_examples = len(labels)

        # Forward each example through backbone + BTRM head
        all_scores = []
        for i in range(n_examples):
            hidden = self._run_backbone_hidden(
                tensors[f"latent_{i}"], tensors[f"sigma_{i}"],
                tensors[f"conditioning_{i}"],
                multiplier=multiplier, attention_backend=attn_backend,
            )
            scores = self._btrm_head(hidden)  # (1, N_heads)
            all_scores.append(scores.squeeze(0))  # (N_heads,)

        all_scores = torch.stack(all_scores)  # (N, N_heads)

        # Split by head and polarity for all-pairs BT loss
        n_heads = self._btrm_head.n_heads
        total_bt = all_scores.new_zeros(())
        total_logsq = all_scores.new_zeros(())
        active_heads = 0
        per_head_accuracy = {}

        for head_idx in range(n_heads):
            pos_mask = torch.tensor(
                [l["head_idx"] == head_idx and l["is_positive"] for l in labels],
                device=self.device,
            )
            neg_mask = torch.tensor(
                [l["head_idx"] == head_idx and not l["is_positive"] for l in labels],
                device=self.device,
            )

            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue

            active_heads += 1
            pos_scores = all_scores[pos_mask, head_idx]
            neg_scores = all_scores[neg_mask, head_idx]

            bt = bt_loss_allpairs(pos_scores, neg_scores)
            total_bt = total_bt + bt

            # Accuracy: fraction of (pos, neg) pairs where pos > neg
            with torch.no_grad():
                diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
                acc = (diff > 0).float().mean().item()
                head_name = self._btrm_head.head_names[head_idx]
                per_head_accuracy[head_name] = acc

            # Logsquare regularization on positives
            if logsq_weight > 0:
                total_logsq = total_logsq + logsquare_regularizer(pos_scores)

        if active_heads > 0:
            total_bt = total_bt / active_heads
            total_logsq = total_logsq / active_heads

        loss = total_bt + logsq_weight * total_logsq

        self._btrm_optimizer.zero_grad()
        loss.backward()
        self._btrm_optimizer.step()

        print(f"  [train_btrm_step] loss={loss.item():.4f} bt={total_bt.item():.4f} "
              f"logsq={total_logsq.item():.4f} heads={active_heads} n={n_examples}")

        return pack_response("ok", metadata={
            "loss": loss.item(),
            "bt_loss": total_bt.item(),
            "logsq_loss": total_logsq.item(),
            "per_head_accuracy": per_head_accuracy,
            "n_examples": n_examples,
            "active_heads": active_heads,
        })

    def handle_accumulate_policy_gradients(self, params, tensors):
        """Compute LoRA parameter gradients and accumulate them on server.

        Same computation as the old compute_policy_gradients but gradients
        stay on server (accumulated in .grad). No tensor output.

        Params:
            adapter_name (str): Which LoRA adapter to differentiate.
            sparse_steps (list[int]): Step indices to evaluate.
            advantage (float): Advantage weight for this rollout.
            multiplier (float, optional): Default 1.0.

        Tensors:
            checkpoint_N: (1, 16, H, W) latent at step N.
            sigmas: (n_steps+1,) full sigma schedule.
            conditioning: (1, seq, dim) text conditioning (positive only).

        Returns:
            Metadata: {total_log_ratio: float, n_steps: int}
        """
        from .lora import set_lora_scale, clear_lora_scale, get_lora_params
        from .sampling import const_calculate_denoised
        from .training_utils import forward_checkpointed

        self._ensure_diffusion()

        adapter_name = params["adapter_name"]
        sparse_steps = params["sparse_steps"]
        advantage = params["advantage"]
        multiplier = params.get("multiplier", 1.0)

        conditioning = tensors["conditioning"].to(device=self.device, dtype=self.dtype)
        sigmas = tensors["sigmas"].to(device=self.device, dtype=self.dtype)

        model = self._diff_model

        # Enable gradients on target LoRA params (don't zero -- accumulate across calls)
        lora_params = list(get_lora_params(model, adapter_name=adapter_name))
        for p in lora_params:
            p.requires_grad_(True)

        # Concurrent batch scale: [pi=1.0, ref=0.0]
        set_lora_scale(
            model,
            torch.tensor([1.0, 0.0], device=self.device, dtype=self.dtype),
            adapter_name=adapter_name,
        )

        # RoPE cache
        first_key = f"checkpoint_{sparse_steps[0]}"
        sample_latent = tensors[first_key]
        B, C, H, W = sample_latent.shape
        num_tokens = conditioning.shape[1]

        padded_h = H + ((-H) % model.patch_size)
        padded_w = W + ((-W) % model.patch_size)
        rope_cache = model.prepare_rope_cache(
            padded_h, padded_w, num_tokens, self.device
        )

        cond_batch = conditioning.expand(2, -1, -1)

        total_log_ratio = 0.0

        for step_idx in sparse_steps:
            x_t = tensors[f"checkpoint_{step_idx}"].to(
                device=self.device, dtype=self.dtype
            )
            sigma = sigmas[step_idx]
            timestep = sigma * multiplier

            x_batch = x_t.expand(2, -1, -1, -1)
            t_batch = timestep.unsqueeze(0).expand(2)

            output_batch, _ = forward_checkpointed(
                model, x_batch, t_batch, cond_batch, num_tokens, rope_cache,
            )

            pi_output, ref_output = output_batch.chunk(2, dim=0)
            pi_denoised = const_calculate_denoised(sigma, pi_output, x_t)
            ref_denoised = const_calculate_denoised(
                sigma, ref_output.detach(), x_t
            )

            diff = pi_denoised - ref_denoised
            mse = (diff * diff).sum()
            log_ratio = -mse / (2.0 * sigma * sigma + 1e-10)

            total_log_ratio += log_ratio.detach().item()

            step_loss = -advantage * log_ratio
            step_loss.backward()

            del x_t, x_batch, output_batch, pi_output, ref_output
            del pi_denoised, ref_denoised, diff

        # Clear concurrent batch scale
        clear_lora_scale(model, adapter_name=adapter_name)

        torch.cuda.empty_cache()

        return pack_response("ok", metadata={
            "total_log_ratio": total_log_ratio,
            "n_steps": len(sparse_steps),
        })

    def handle_policy_optimizer_step(self, params, tensors):
        """Clip gradients, step optimizer, zero gradients.

        Params:
            adapter_name (str): Which LoRA adapter to step.
            max_grad_norm (float, optional): Default 1.0.
            lr (float, optional): Default 1e-4.

        Returns:
            Metadata: {grad_norm: float, n_params: int}
        """
        from .lora import get_lora_params

        self._ensure_diffusion()

        adapter_name = params["adapter_name"]
        max_grad_norm = params.get("max_grad_norm", 1.0)
        lr = params.get("lr", 1e-4)

        model = self._diff_model
        lora_params = list(get_lora_params(model, adapter_name=adapter_name))

        # Lazy-init optimizer on first call
        if adapter_name not in self._policy_optimizers:
            self._policy_optimizers[adapter_name] = torch.optim.AdamW(
                lora_params, lr=lr,
            )
            print(f"  [policy_optimizer_step] Created optimizer for {adapter_name} "
                  f"({len(lora_params)} params, lr={lr})")

        optimizer = self._policy_optimizers[adapter_name]

        # Clip and step
        grad_norm = torch.nn.utils.clip_grad_norm_(lora_params, max_grad_norm)
        optimizer.step()

        # Zero gradients for next accumulation cycle
        for p in lora_params:
            if p.grad is not None:
                p.grad = None

        grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        print(f"  [policy_optimizer_step] {adapter_name}: grad_norm={grad_norm_val:.3e}")

        return pack_response("ok", metadata={
            "grad_norm": grad_norm_val,
            "n_params": len(lora_params),
        })

    def handle_get_lora_state_dict(self, params, tensors):
        """Retrieve current LoRA weights from the server.

        Params:
            adapter_name (str, optional): Filter by adapter name.

        Returns:
            Tensors: LoRA weight tensors with keys "path.adapters.name.lora_A" etc.
            Metadata: {"n_tensors": int}
        """
        from .lora import lora_state_dict

        self._ensure_diffusion()

        adapter_name = params.get("adapter_name")
        sd = lora_state_dict(self._diff_model, adapter_name=adapter_name)

        return pack_response("ok", sd, {"n_tensors": len(sd)})

    def handle_free(self, params, tensors):
        """Free specified model(s).

        Params:
            model (str): "te", "diffusion", "vae", or "all".
        """
        target = params.get("model", "all")
        if target == "all":
            self._free_all()
        elif target == "te":
            self._free_te()
        elif target == "diffusion":
            if self._diff_model is not None:
                self._snapshot_lora_weights()
                del self._diff_model, self._diff_compiled, self._diff_compiled_packed
                self._diff_model = None
                self._diff_compiled = None
                self._diff_compiled_packed = None
                torch.cuda.empty_cache()
        elif target == "vae":
            if self._vae_model is not None:
                del self._vae_model
                self._vae_model = None
                torch.cuda.empty_cache()
        print(f"  [lifecycle] Freed {target}")
        return pack_response("ok")

    def handle_dump_all_loras(self, params, tensors):
        """Emergency dump: save every LoRA adapter to disk as safetensors.

        Enumerates all adapter names on the diffusion model, saves each
        to a separate file.  Designed for crash recovery — call this when
        the client is dying, the process is about to exit, or you just
        want a snapshot of all active LoRA state.

        Params:
            output_dir (str, optional): Directory for dump files.
                Defaults to "lora_dumps" in the working directory.

        Returns:
            Metadata: {"files": [...], "manifest": str}
        """
        from pathlib import Path
        from safetensors.torch import save_file as st_save
        from .lora import enumerate_adapters, lora_state_dict

        output_dir = Path(params.get("output_dir", "lora_dumps"))
        output_dir.mkdir(parents=True, exist_ok=True)

        if self._diff_model is None:
            return pack_response("ok", metadata={
                "files": [], "note": "no diffusion model loaded"})

        adapters = enumerate_adapters(self._diff_model)
        if not adapters:
            return pack_response("ok", metadata={
                "files": [], "note": "no adapters found"})

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        files = []
        for adapter_name, info in adapters.items():
            sd = lora_state_dict(self._diff_model, adapter_name=adapter_name)
            sd_cpu = {k: v.cpu() for k, v in sd.items()}
            fname = f"{adapter_name}_{timestamp}.safetensors"
            fpath = output_dir / fname
            st_save(sd_cpu, str(fpath))
            files.append({
                "adapter": adapter_name,
                "path": str(fpath),
                "n_tensors": len(sd_cpu),
                "rank": info["rank"],
                "alpha": info["alpha"],
                "scale": info["scale"],
            })
            print(f"  [dump_all_loras] {adapter_name}: "
                  f"{len(sd_cpu)} tensors -> {fpath}")

        # Also dump BTRM head if present
        btrm_file = None
        if self._btrm_head is not None:
            btrm_sd = {k: v.cpu() for k, v in self._btrm_head.state_dict().items()}
            btrm_fname = f"btrm_head_{timestamp}.safetensors"
            btrm_fpath = output_dir / btrm_fname
            st_save(btrm_sd, str(btrm_fpath))
            btrm_file = {
                "path": str(btrm_fpath),
                "n_tensors": len(btrm_sd),
                "config": self._btrm_config,
            }
            print(f"  [dump_all_loras] BTRM head: {len(btrm_sd)} tensors -> {btrm_fpath}")

        import json as _json
        manifest_path = output_dir / f"dump_manifest_{timestamp}.json"
        manifest = {"timestamp": timestamp, "adapters": files}
        if btrm_file is not None:
            manifest["btrm_head"] = btrm_file
        with open(manifest_path, "w") as f:
            _json.dump(manifest, f, indent=2)

        print(f"  [dump_all_loras] {len(files)} adapter(s) dumped to {output_dir}")
        return pack_response("ok", metadata={
            "files": files, "manifest": str(manifest_path),
            "btrm_head": btrm_file,
        })

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    _HANDLERS = {
        "encode_prompt": "handle_encode_prompt",
        "sample_trajectory": "handle_sample_trajectory",
        "sample_trajectory_packed": "handle_sample_trajectory_packed",
        "vae_encode": "handle_vae_encode",
        "vae_decode": "handle_vae_decode",
        "warmup": "handle_warmup",
        "warmup_packed": "handle_warmup_packed",
        "status": "handle_status",
        "free": "handle_free",
        "inject_lora": "handle_inject_lora",
        "update_lora_weights": "handle_update_lora_weights",
        "set_adapter_config": "handle_set_adapter_config",
        "get_lora_state_dict": "handle_get_lora_state_dict",
        "dump_all_loras": "handle_dump_all_loras",
        "inject_btrm_head": "handle_inject_btrm_head",
        "score_btrm": "handle_score_btrm",
        "train_btrm_step": "handle_train_btrm_step",
        "accumulate_policy_gradients": "handle_accumulate_policy_gradients",
        "policy_optimizer_step": "handle_policy_optimizer_step",
    }

    def dispatch(self, method, params, tensors):
        handler_name = self._HANDLERS.get(method)
        if handler_name is None:
            return pack_response("error", metadata={"error": f"Unknown method: {method}"})
        handler = getattr(self, handler_name)
        return handler(params, tensors)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def serve(self, endpoint: str = "tcp://*:5555"):
        """Run the server main loop."""
        ctx = zmq.Context()
        socket = ctx.socket(zmq.REP)
        socket.bind(endpoint)
        print(f"Inference server listening on {endpoint}")
        print(f"  Models: diff={self.fp8_diff_path}")
        print(f"          te={self.te_path}")
        print(f"          vae={self.vae_path}")
        print(f"  Device: {self.device}, dtype: {self.dtype}")

        while True:
            try:
                frames = socket.recv_multipart()
                method, params, tensors = unpack_request(frames)
                print(f"  [{method}] ...", end="", flush=True)
                t0 = time.perf_counter()

                response_frames = self.dispatch(method, params, tensors)

                elapsed = time.perf_counter() - t0
                print(f" {elapsed:.2f}s")
                socket.send_multipart(response_frames)

            except KeyboardInterrupt:
                print("\nShutting down...")
                break
            except Exception:
                tb = traceback.format_exc()
                print(f"\n  ERROR:\n{tb}")
                try:
                    socket.send_multipart(
                        pack_response("error", metadata={"error": tb})
                    )
                except Exception:
                    pass  # socket may be in bad state

        socket.close()
        ctx.term()
        self._free_all()


def main():
    parser = argparse.ArgumentParser(
        description="futudiffu inference server (ZeroMQ)")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--fp8-diff", required=True,
                        help="Path to FP8 blockwise diffusion model safetensors")
    parser.add_argument("--te", required=True,
                        help="Path to text encoder safetensors")
    parser.add_argument("--vae", required=True,
                        help="Path to VAE safetensors")
    parser.add_argument("--tokenizer", default=None,
                        help="Path to tokenizer directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    server = InferenceServer(
        fp8_diff_path=args.fp8_diff,
        te_path=args.te,
        vae_path=args.vae,
        tokenizer_path=args.tokenizer,
        device=args.device,
        dtype=args.dtype,
    )
    server.serve(f"tcp://*:{args.port}")


if __name__ == "__main__":
    main()

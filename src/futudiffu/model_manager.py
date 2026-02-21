"""Model lifecycle manager: load/free/warmup for TE, diffusion, VAE.

Owns all model state and VRAM lifecycle. The server delegates to this
class for model loading, LoRA replay, compilation, and VRAM management.

Model groups are mutually exclusive for large allocations:
  - "te" phase: text encoder (~7.5GB)
  - "diffusion" phase: FP8 diffusion model (~8GB with compile) + optionally VAE
  - "vae" phase: VAE only (~320MB)
"""

import time

import torch


class ModelManager:
    """Manages GPU model lifecycle: load, free, compile, LoRA replay."""

    def __init__(
        self,
        fp8_diff_path: str,
        te_path: str,
        vae_path: str,
        tokenizer_path: str | None,
        device: torch.device,
        dtype: torch.dtype,
        fp8_block_size: int = 128,
    ):
        self.fp8_diff_path = fp8_diff_path
        self.te_path = te_path
        self.vae_path = vae_path
        self.tokenizer_path = tokenizer_path
        self.device = device
        self.dtype = dtype
        self.fp8_block_size = fp8_block_size

        # Model state -- at most one large model loaded at a time
        self.te_model = None
        self.tokenizer = None
        self.diff_model = None         # raw model
        self.diff_compiled = None       # torch.compile'd forward()
        self.diff_compiled_packed = None  # torch.compile'd forward_packed()
        self.vae_model = None

        # Which phase we're in determines what's loaded
        self.phase = None  # "te" | "diffusion" | "vae" | None

        # Sage configured once at startup
        self.sage_configured = False

        # LoRA lifecycle: survive model swaps (TE <-> diffusion)
        self.lora_configs: list[dict] = []  # Injection configs to replay on reload
        self.lora_weights: dict[str, dict[str, torch.Tensor]] = {}  # CPU weights per adapter
        self.lora_scales: dict[str, float | list[float]] = {}  # Last-set scale per adapter

        # BTRM score unembedder: lives alongside LoRA, persisted together (~30KB, permanent GPU resident)
        self.btrm_head: torch.nn.Module | None = None
        self.btrm_optimizer: torch.optim.Optimizer | None = None
        self.btrm_config: dict | None = None  # For persistence/crash dump

        # Policy optimizer: lazy-init on first policy_optimizer_step call
        self.policy_optimizers: dict[str, torch.optim.Optimizer] = {}

    # ------------------------------------------------------------------
    # LoRA weight snapshotting
    # ------------------------------------------------------------------

    def snapshot_lora_weights(self):
        """Save current LoRA weights to CPU before destroying diffusion model."""
        if self.diff_model is not None and self.lora_configs:
            from .lora import lora_state_dict
            for cfg in self.lora_configs:
                name = cfg["adapter_name"]
                sd = lora_state_dict(self.diff_model, adapter_name=name)
                self.lora_weights[name] = {
                    k: v.detach().cpu() for k, v in sd.items()
                }

    # ------------------------------------------------------------------
    # Free
    # ------------------------------------------------------------------

    def free_all(self):
        """Free all models from VRAM."""
        if self.te_model is not None:
            del self.te_model
            self.te_model = None
            self.tokenizer = None
        if self.diff_model is not None:
            self.snapshot_lora_weights()
            del self.diff_model, self.diff_compiled, self.diff_compiled_packed
            self.diff_model = None
            self.diff_compiled = None
            self.diff_compiled_packed = None
        if self.vae_model is not None:
            del self.vae_model
            self.vae_model = None
        self.phase = None
        torch.cuda.empty_cache()

    def free_te(self):
        """Free text encoder."""
        if self.te_model is not None:
            del self.te_model
            self.te_model = None
            self.tokenizer = None
            torch.cuda.empty_cache()

    def free_diffusion(self):
        """Free diffusion model, snapshotting LoRA weights first."""
        if self.diff_model is not None:
            self.snapshot_lora_weights()
            del self.diff_model, self.diff_compiled, self.diff_compiled_packed
            self.diff_model = None
            self.diff_compiled = None
            self.diff_compiled_packed = None
            torch.cuda.empty_cache()

    def free_vae(self):
        """Free VAE model."""
        if self.vae_model is not None:
            del self.vae_model
            self.vae_model = None
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Ensure (load if needed)
    # ------------------------------------------------------------------

    def ensure_te(self):
        """Ensure text encoder is loaded. Frees other large models if needed."""
        if self.te_model is not None:
            return
        # Free diffusion (large) but keep VAE (small, can coexist)
        if self.diff_model is not None:
            self.snapshot_lora_weights()
            del self.diff_model, self.diff_compiled, self.diff_compiled_packed
            self.diff_model = None
            self.diff_compiled = None
            self.diff_compiled_packed = None
            torch.cuda.empty_cache()

        from .text_encoder import create_tokenizer, load_text_encoder

        self.tokenizer = create_tokenizer(self.tokenizer_path)
        self.te_model = load_text_encoder(
            self.te_path, device=self.device, dtype=self.dtype
        )
        self.te_model = torch.compile(self.te_model, mode="default")
        self.phase = "te"
        print(f"  [lifecycle] TE loaded ({self.dtype})")

    def ensure_diffusion(self):
        """Ensure diffusion model is loaded. Frees TE if needed."""
        if self.diff_model is not None:
            return
        # Free TE (large)
        self.free_te()

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
        self.diff_model = model

        # Replay LoRA injections that were active before the lifecycle swap.
        # fuse_model() must run first because LoRA targets post-fusion modules
        # (e.g. the fused w1w3 FP8Linear).
        if self.lora_configs:
            from .lora import inject_lora, load_lora_state_dict, set_lora_scale
            for cfg in self.lora_configs:
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
            for adapter_name, weights in self.lora_weights.items():
                if weights:
                    # Move weights to model dtype for load
                    sd = {k: v.to(dtype=self.dtype) for k, v in weights.items()}
                    load_lora_state_dict(model, sd)
            # Restore scale settings
            for adapter_name, scale in self.lora_scales.items():
                if isinstance(scale, (int, float)):
                    scale_tensor = torch.tensor(
                        [scale], device=self.device, dtype=self.dtype)
                else:
                    scale_tensor = torch.tensor(
                        scale, device=self.device, dtype=self.dtype)
                set_lora_scale(model, scale_tensor, adapter_name=adapter_name)
            print(f"  [lifecycle] Replayed {len(self.lora_configs)} LoRA injection(s)")

        # torch.compile after LoRA replay so the compiled graph includes
        # LoRA modules from the start, avoiding an extra dynamo reset.
        self.diff_compiled = torch.compile(model, mode="default")
        # forward_packed is a separate method -- torch.compile(model) only
        # wraps forward(). Compile it separately for FlexAttention perf.
        self.diff_compiled_packed = torch.compile(
            model.forward_packed, mode="default"
        )
        self.phase = "diffusion"
        elapsed = time.perf_counter() - t0
        print(f"  [lifecycle] Diffusion loaded (FP8, fused, compiled) in {elapsed:.1f}s")

    def ensure_vae(self):
        """Ensure VAE is loaded. VAE is small enough to coexist with diffusion."""
        if self.vae_model is not None:
            return
        from .vae import load_vae

        self.vae_model = load_vae(
            self.vae_path, device=self.device, dtype=self.dtype
        )
        print(f"  [lifecycle] VAE loaded")

    # ------------------------------------------------------------------
    # Sage attention configuration
    # ------------------------------------------------------------------

    def configure_sage_if_needed(self, attention_backend: str):
        """Configure sage attention on first use. Idempotent."""
        if not self.sage_configured and attention_backend != "sdpa":
            from .sage_attention import configure_sage
            configure_sage(smooth_k=True, qk_quant="int8", pv_quant="bf16")
            self.sage_configured = True

    # ------------------------------------------------------------------
    # Per-layer compilation for training (gradient checkpointing compatible)
    # ------------------------------------------------------------------

    def compile_layers_for_training(self) -> list:
        """Compile individual transformer layers for gradient-checkpointed training.

        Unlike whole-model compile (which creates one graph for model.forward()),
        this compiles each of the 30 main transformer layers independently.
        This is compatible with torch.utils.checkpoint which needs to call
        layers individually within grad_ckpt().

        Falls back gracefully: if any layer fails to compile (e.g., SymPy
        recursion with LoRA on torch 2.10.0), that layer runs eager while
        others stay compiled. suppress_errors=True means dynamo doesn't raise;
        it just falls back to eager for that layer.

        Returns:
            List of compiled (or fallback-eager) layer callables.
            Also stored as self._compiled_training_layers for reuse.
        """
        if self.diff_model is None:
            return []
        import torch._dynamo
        old_suppress = torch._dynamo.config.suppress_errors
        torch._dynamo.config.suppress_errors = True
        compiled = []
        n_compiled = 0
        try:
            for i, layer in enumerate(self.diff_model.layers):
                try:
                    c = torch.compile(layer, mode="reduce-overhead")
                    compiled.append(c)
                    n_compiled += 1
                except Exception as e:
                    print(f"  [compile_layers] Layer {i} compile failed: {e}")
                    compiled.append(layer)
        finally:
            torch._dynamo.config.suppress_errors = old_suppress
        self._compiled_training_layers = compiled
        print(f"  [compile_layers] {n_compiled}/{len(compiled)} layers compiled for training")
        return compiled

    def get_compiled_training_layers(self) -> list | None:
        """Return compiled layers if available, None otherwise."""
        return getattr(self, '_compiled_training_layers', None)

    # ------------------------------------------------------------------
    # Compilation reset (after LoRA injection)
    # ------------------------------------------------------------------

    def recompile_diffusion(self):
        """Reset dynamo and recompile diffusion model (after LoRA injection)."""
        torch._dynamo.reset()
        self.diff_compiled = torch.compile(self.diff_model, mode="default")
        self.diff_compiled_packed = torch.compile(
            self.diff_model.forward_packed, mode="default"
        )

    # ------------------------------------------------------------------
    # Server-facing RPC helpers (extracted from server.py handlers)
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return server status: loaded models, VRAM stats."""
        vram_allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
        vram_reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
        vram_total = torch.cuda.get_device_properties(
            self.device).total_memory / (1024**3)

        loaded = []
        if self.te_model is not None:
            loaded.append("te")
        if self.diff_model is not None:
            loaded.append("diffusion")
        if self.vae_model is not None:
            loaded.append("vae")

        return {
            "loaded_models": loaded,
            "phase": self.phase,
            "vram_allocated_gb": round(vram_allocated, 2),
            "vram_reserved_gb": round(vram_reserved, 2),
            "vram_total_gb": round(vram_total, 2),
            "sage_configured": self.sage_configured,
        }

    def allocate_adapter_rpc(self, params: dict) -> dict:
        """Allocate adapter slots — graph-mutating, NO recompile.

        Call this for all adapters BEFORE compile_and_warmup(). The adapter
        starts silent (scale=0, zero weights). Use set_adapter_config to
        activate and init_adapter_weights to initialize for training.
        """
        from .lora import allocate_adapter

        adapter_name = params["adapter_name"]
        rank = params.get("rank", 8)
        alpha = params.get("alpha", 16.0)
        layer_indices = params.get("layer_indices")
        if layer_indices is not None:
            layer_indices = set(layer_indices)

        injected = allocate_adapter(
            self.diff_model,
            name=adapter_name,
            rank=rank,
            alpha=alpha,
            layer_indices=layer_indices,
        )

        self.lora_configs.append({
            "adapter_name": adapter_name,
            "rank": rank,
            "alpha": alpha,
            "layer_indices": layer_indices,
            "init_b_std": 0.0,
        })

        n_params = sum(
            a.lora_A.numel() + a.lora_B.numel() for a in injected.values()
        )
        return {
            "adapter_name": adapter_name,
            "n_adapters": len(injected),
            "n_params": n_params,
            "graph_mutated": True,
        }

    def init_adapter_weights_rpc(self, params: dict) -> dict:
        """(Re-)initialize adapter weights — graph-invariant, safe after compile."""
        from .lora import init_adapter_weights

        adapter_name = params["adapter_name"]
        init_b_std = params.get("init_b_std", 0.0)
        scale = params.get("scale", 1.0)

        n = init_adapter_weights(
            self.diff_model,
            name=adapter_name,
            init_b_std=init_b_std,
            scale=scale,
        )
        return {
            "adapter_name": adapter_name,
            "n_modules_initialized": n,
        }

    def inject_lora_adapter(self, params: dict) -> dict:
        """Legacy: allocate + init + recompile in one call.

        Prefer allocate_adapter_rpc() + init_adapter_weights_rpc() for new
        code to avoid mid-session recompilation.
        """
        from .lora import inject_lora

        adapter_name = params["adapter_name"]
        rank = params.get("rank", 8)
        alpha = params.get("alpha", 16.0)
        layer_indices = params.get("layer_indices")
        init_b_std = params.get("init_b_std", 0.0)
        if layer_indices is not None:
            layer_indices = set(layer_indices)

        injected = inject_lora(
            self.diff_model,
            name=adapter_name,
            rank=rank,
            alpha=alpha,
            layer_indices=layer_indices,
            init_b_std=init_b_std,
        )

        self.lora_configs.append({
            "adapter_name": adapter_name,
            "rank": rank,
            "alpha": alpha,
            "layer_indices": layer_indices,
            "init_b_std": init_b_std,
        })

        self.recompile_diffusion()

        n_params = sum(
            a.lora_A.numel() + a.lora_B.numel() for a in injected.values()
        )
        return {
            "adapter_name": adapter_name,
            "n_adapters": len(injected),
            "n_params": n_params,
        }

    def update_lora_weights_rpc(self, tensors: dict) -> dict:
        """Load LoRA weights in-place + update CPU cache. Returns metadata."""
        from .lora import load_lora_state_dict

        sd = {k: v.to(dtype=self.dtype) for k, v in tensors.items()}
        load_lora_state_dict(self.diff_model, sd)

        for key, val in sd.items():
            parts = key.split(".adapters.")
            if len(parts) == 2:
                adapter_name = parts[1].split(".")[0]
                if adapter_name not in self.lora_weights:
                    self.lora_weights[adapter_name] = {}
                self.lora_weights[adapter_name][key] = val.detach().cpu()

        return {"n_tensors": len(sd)}

    def set_adapter_config_rpc(self, params: dict) -> dict:
        """Set adapter scale/freeze + persistence. Returns metadata."""
        from .lora import (
            clear_lora_scale, freeze_adapter, set_lora_scale, unfreeze_adapter,
        )

        adapter_name = params["adapter_name"]
        scale = params.get("scale")
        frozen = params.get("frozen")

        if scale is not None:
            if isinstance(scale, (int, float)):
                scale_tensor = torch.tensor(
                    [scale], device=self.device, dtype=self.dtype)
            else:
                scale_tensor = torch.tensor(
                    scale, device=self.device, dtype=self.dtype)
            set_lora_scale(
                self.diff_model, scale_tensor, adapter_name=adapter_name)
            self.lora_scales[adapter_name] = scale
        elif scale is None and "scale" in params:
            clear_lora_scale(self.diff_model, adapter_name=adapter_name)
            self.lora_scales.pop(adapter_name, None)

        n_frozen = 0
        if frozen is True:
            n_frozen = freeze_adapter(self.diff_model, adapter_name)
        elif frozen is False:
            n_frozen = unfreeze_adapter(self.diff_model, adapter_name)

        return {
            "adapter_name": adapter_name,
            "n_frozen": n_frozen,
        }

    def inject_btrm_head_rpc(self, params: dict) -> dict:
        """Create BTRM head + optimizer + config. Returns metadata."""
        from .btrm import ScoreUnembedder

        hidden_dim = params.get("hidden_dim", 3840)
        head_names = params.get("head_names", ["bit_quality", "step_quality"])
        logit_cap = params.get("logit_cap", 10.0)
        lr = params.get("lr")
        weight_decay = params.get("weight_decay", 0.0)

        self.btrm_head = ScoreUnembedder(
            hidden_dim=hidden_dim,
            head_names=head_names,
            logit_cap=logit_cap,
        ).to(device=self.device, dtype=self.dtype)
        self.btrm_head.train()

        self.btrm_config = {
            "hidden_dim": hidden_dim,
            "head_names": list(head_names),
            "logit_cap": logit_cap,
        }

        if lr is not None:
            # Include rtheta LoRA adapter params in the optimizer.
            # Run 2 fix for Defect 24: optimizer must include adapter parameters
            # so that rtheta LoRA receives gradient updates during BTRM training.
            from .lora import get_lora_params
            rtheta_params = list(get_lora_params(self.diff_model, adapter_name="rtheta")) \
                if self.diff_model is not None else []

            param_groups = [
                {"params": list(self.btrm_head.parameters()), "lr": lr},
            ]
            if rtheta_params:
                param_groups.append({"params": rtheta_params, "lr": lr})

            self.btrm_optimizer = torch.optim.AdamW(
                param_groups, weight_decay=weight_decay,
            )

        n_params = sum(p.numel() for p in self.btrm_head.parameters())
        return {
            "n_heads": len(head_names),
            "n_params": n_params,
            "has_optimizer": lr is not None,
        }

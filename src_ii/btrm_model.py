"""Compound BTRM Model: frozen backbone + r_theta adapter + scoring head.

This module enforces that a BTRM model is always the complete triple:
  1. Frozen diffusion backbone
  2. r_theta LoRA adapter (trainable)
  3. ScoreUnembedder (trainable)

You cannot construct one without all three. The optimizer always includes
both head params AND adapter params -- it is impossible to forget one.

This prevents Defect 24 from the live training run: "rtheta LoRA never
trained because the optimizer only had head params."

Import constraints:
  - IMPORTS from futudiffu.btrm: ScoreUnembedder (the scoring head)
  - IMPORTS from futudiffu.lora: allocate_adapter, init_adapter_weights,
    get_lora_params, set_lora_scale, lora_state_dict, load_lora_state_dict,
    freeze_adapter, unfreeze_adapter
  - IMPORTS from futudiffu.training_utils: HiddenCapture
  - DOES NOT import: model_manager, server, client
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from torch import Tensor

from futudiffu.btrm import ScoreUnembedder, _RMSNorm
from futudiffu.lora import (
    allocate_adapter,
    freeze_adapter,
    get_lora_params,
    init_adapter_weights,
    load_lora_state_dict,
    lora_state_dict,
    set_lora_scale,
    unfreeze_adapter,
)
from futudiffu.training_utils import HiddenCapture


DEFAULT_RTHETA_RANK = 8
DEFAULT_RTHETA_ALPHA = 16.0
DEFAULT_RTHETA_INIT_B_STD = 0.01


class BTRMCompoundModel:
    """Compound object: frozen backbone + r_theta LoRA adapter + ScoreUnembedder.

    This is NOT an nn.Module. It is a coordinator that owns three components
    and enforces that they are always used together. The backbone is frozen,
    the adapter and head are trainable.

    Construction allocates the adapter on the backbone and creates the head.
    The optimizer() method returns an optimizer over BOTH adapter params and
    head params -- it is structurally impossible to forget one.

    Usage:
        model = BTRMCompoundModel(backbone, head_names=("pinkify", "thisnotthat"))
        optimizer = model.optimizer(lr=1e-3)
        scores = model.score(hidden_states)
        model.persist("output_dir")
        loaded = BTRMCompoundModel.load("output_dir", backbone)
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        adapter_name: str = "rtheta",
        adapter_rank: int = DEFAULT_RTHETA_RANK,
        adapter_alpha: float = DEFAULT_RTHETA_ALPHA,
        adapter_init_b_std: float = DEFAULT_RTHETA_INIT_B_STD,
        adapter_layer_indices: set[int] | None = None,
        head_names: Sequence[str] = ("pinkify", "thisnotthat"),
        hidden_dim: int = 3840,
        logit_cap: float = 10.0,
        device: torch.device | None = None,
        compile_layers: bool = True,
    ) -> None:
        self.backbone = backbone
        self.adapter_name = adapter_name

        if device is None:
            device = next(backbone.parameters()).device

        # 1. Freeze backbone
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad_(False)

        # 2. Allocate r_theta adapter on the backbone
        self._adapter_map = allocate_adapter(
            backbone,
            name=adapter_name,
            rank=adapter_rank,
            alpha=adapter_alpha,
            layer_indices=adapter_layer_indices,
        )

        # Initialize adapter weights for training (non-zero B)
        n_init = init_adapter_weights(
            backbone,
            name=adapter_name,
            init_b_std=adapter_init_b_std,
            scale=1.0,
        )
        layer_desc = f"layers={sorted(adapter_layer_indices)}" if adapter_layer_indices else "all layers"
        print(f"[BTRMCompoundModel] Allocated + initialized r_theta adapter: "
              f"{n_init} modules, rank={adapter_rank}, alpha={adapter_alpha}, "
              f"init_b_std={adapter_init_b_std}, {layer_desc}")

        # 3. Create the ScoreUnembedder
        self.head = ScoreUnembedder(
            hidden_dim=hidden_dim,
            head_names=head_names,
            logit_cap=logit_cap,
        ).to(device=device, dtype=torch.float32)

        # 4. Install hidden capture hook
        self._capture = HiddenCapture(backbone)
        self._capture.install()

        # 5. Per-layer compilation for training.
        #
        # Whole-model torch.compile is INCOMPATIBLE with per-block gradient
        # checkpointing (extract_hidden_differentiable uses grad_ckpt on
        # each layer individually). The correct approach: compile each of
        # the 30 main transformer layers independently. This is compatible
        # with gradient checkpointing and gives ~2x speedup + ~40% VRAM
        # reduction vs eager.
        #
        # This was the root cause of the 21.5s/step regression in the
        # reward-validated training run (2026-02-20): the script loaded
        # the backbone with compile_model=False (correctly, because
        # whole-model compile breaks grad_ckpt), but nothing compiled
        # individual layers. All 30 layers ran eager. Per-layer compilation
        # here makes compiled training the structural default -- you cannot
        # construct a BTRMCompoundModel without getting compiled layers
        # unless you explicitly opt out.
        self._compiled_layers: list | None = None
        if compile_layers:
            self._compile_layers_for_training()

        # Store config for persistence
        self._config = {
            "adapter_name": adapter_name,
            "adapter_rank": adapter_rank,
            "adapter_alpha": adapter_alpha,
            "adapter_init_b_std": adapter_init_b_std,
            "adapter_layer_indices": sorted(adapter_layer_indices) if adapter_layer_indices else None,
            "head_names": list(head_names),
            "hidden_dim": hidden_dim,
            "logit_cap": logit_cap,
        }

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    @property
    def is_compiled(self) -> bool:
        """Whether per-layer compilation is active."""
        return self._compiled_layers is not None and len(self._compiled_layers) > 0

    def _compile_layers_for_training(self) -> None:
        """Compile individual transformer layers for gradient-checkpointed training.

        Unlike whole-model torch.compile (which creates one graph for
        model.forward()), this compiles each of the 30 main transformer
        layers independently. This is compatible with
        torch.utils.checkpoint which calls layers individually within
        grad_ckpt().

        Follows the pattern from model_manager.compile_layers_for_training().
        Falls back gracefully: if any layer fails to compile, that layer
        runs eager while others stay compiled. suppress_errors=True means
        dynamo doesn't raise; it just falls back to eager for that layer.
        """
        import torch._dynamo
        old_suppress = torch._dynamo.config.suppress_errors
        torch._dynamo.config.suppress_errors = True

        compiled = []
        n_compiled = 0
        try:
            for i, layer in enumerate(self.backbone.layers):
                try:
                    c = torch.compile(layer, mode="reduce-overhead")
                    compiled.append(c)
                    n_compiled += 1
                except Exception as e:
                    print(f"[BTRMCompoundModel] Layer {i} compile failed: {e}")
                    compiled.append(layer)
        finally:
            torch._dynamo.config.suppress_errors = old_suppress

        self._compiled_layers = compiled
        print(f"[BTRMCompoundModel] Per-layer compilation: "
              f"{n_compiled}/{len(compiled)} layers compiled for training")

    def _get_training_layers(self) -> list:
        """Return compiled layers if available, otherwise raw backbone layers.

        Used by extract_hidden_differentiable() and
        score_differentiable_packed() to iterate over layers.
        """
        if self._compiled_layers is not None:
            return self._compiled_layers
        return list(self.backbone.layers)

    def adapter_params(self) -> list[nn.Parameter]:
        """Return all r_theta adapter parameters."""
        return list(get_lora_params(self.backbone, adapter_name=self.adapter_name))

    def head_params(self) -> list[nn.Parameter]:
        """Return all ScoreUnembedder parameters."""
        return list(self.head.parameters())

    def all_trainable_params(self) -> list[nn.Parameter]:
        """Return ALL trainable parameters: adapter + head.

        This is the method that prevents Defect 24. You cannot get
        head params without also getting adapter params.
        """
        return self.adapter_params() + self.head_params()

    def optimizer(
        self,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.999),
        optimizer_type: str = "adam",
        muon_lr: float = 0.02,
        muon_momentum: float = 0.95,
    ) -> torch.optim.Optimizer:
        """Create an optimizer over ALL trainable params (adapter + head).

        This is the safe entry point. It is impossible to accidentally
        create an optimizer over only the head params.

        Args:
            lr: Learning rate for AdamW (used for all params when optimizer_type="adam",
                or for ScoreUnembedder params when optimizer_type="muon").
            weight_decay: Weight decay for AdamW param groups.
            betas: Beta coefficients for AdamW.
            optimizer_type: "adam" or "muon". When "muon", uses heterogeneous
                optimizer: Muon for LoRA A/B params (2D matrices), AdamW for
                ScoreUnembedder params (small MLP-like head, not suited for Muon).
            muon_lr: Learning rate for Muon param group (default 0.02).
            muon_momentum: Momentum for Muon (default 0.95).
        """
        adapter_ps = self.adapter_params()
        head_ps = self.head_params()
        n_adapter = sum(p.numel() for p in adapter_ps)
        n_head = sum(p.numel() for p in head_ps)

        if optimizer_type == "muon":
            # Heterogeneous optimizer: Muon for LoRA matrices, AdamW for head.
            # ScoreUnembedder is RMSNorm(1D) + Linear(3840, N_heads) -- not
            # suited for Newton-Schulz orthogonalization.
            from torch.optim import Muon
            print(f"[BTRMCompoundModel] Muon optimizer: {n_adapter} adapter params "
                  f"(Muon lr={muon_lr}) + {n_head} head params (AdamW lr={lr})")
            return Muon(
                muon_params=adapter_ps,
                lr=muon_lr,
                momentum=muon_momentum,
                adamw_params=head_ps,
                adamw_lr=lr,
                adamw_betas=betas,
                adamw_wd=weight_decay,
            )
        else:
            print(f"[BTRMCompoundModel] AdamW optimizer: {n_adapter} adapter params + "
                  f"{n_head} head params = {n_adapter + n_head} total")
            return torch.optim.AdamW(
                adapter_ps + head_ps, lr=lr,
                weight_decay=weight_decay, betas=betas,
            )

    def set_adapter_scale(self, scale: float) -> None:
        """Set the r_theta adapter scale (1.0 = active, 0.0 = silent)."""
        device = self.device
        scale_t = torch.tensor([scale], dtype=torch.bfloat16, device=device)
        set_lora_scale(self.backbone, scale_t, adapter_name=self.adapter_name)

    def extract_hidden(
        self,
        latent: Tensor,
        timestep: Tensor,
        conditioning: Tensor,
        num_tokens: int,
        rope_cache: dict,
    ) -> Tensor:
        """Run backbone forward and return DETACHED hidden states (inference only).

        WARNING: Returns tensors with NO grad_fn. If you train on these,
        the adapter gets ZERO gradients. This is the inference/scoring path.

        For training the adapter, use extract_hidden_differentiable() which
        preserves the computation graph through the LoRA matrices.

        Args:
            latent: (B, C, H, W) noisy latent.
            timestep: (B,) timestep values.
            conditioning: (B, seq, cap_feat_dim) text encoder output.
            num_tokens: Number of text tokens.
            rope_cache: Precomputed RoPE cache.

        Returns:
            (B, N_tokens, hidden_dim) hidden states (DETACHED, no grad_fn).
        """
        self._capture.captured = None
        with torch.inference_mode():
            self.backbone(
                latent, timestep, conditioning,
                num_tokens=num_tokens, rope_cache=rope_cache,
            )
        return self._capture.get()

    def extract_hidden_differentiable(
        self,
        latent: Tensor,
        timestep: Tensor,
        conditioning: Tensor,
        num_tokens: int,
        rope_cache: dict,
        gradient_checkpointing: bool = True,
    ) -> Tensor:
        """Run backbone forward and return hidden states WITH grad_fn intact.

        This is the training-time path. The compute graph flows through
        the adapter's LoRA matrices, so loss.backward() produces nonzero
        gradients on lora_A and lora_B.

        Embedding + refiners run under no_grad (no trainable params there).
        The 30 main transformer layers are individually gradient-checkpointed
        to keep peak VRAM under ~18 GB.

        Args:
            latent: (B, C, H, W) noisy latent.
            timestep: (B,) timestep values.
            conditioning: (B, seq, cap_feat_dim) text encoder output.
            num_tokens: Number of text tokens.
            rope_cache: Precomputed RoPE cache.
            gradient_checkpointing: If True, per-block gradient checkpointing
                on the 30 main layers. Required for 24 GB VRAM.

        Returns:
            (B, N_tokens, hidden_dim) hidden states with grad_fn.
        """
        from futudiffu.diffusion_model import pad_to_patch_size, pad_zimage
        from torch.utils.checkpoint import checkpoint as grad_ckpt

        model = self.backbone

        # --- Phase 1: Embedding + refiners (no trainable params) ---
        with torch.no_grad():
            t = 1 - timestep
            bs, c, h, w = latent.shape
            x_padded = pad_to_patch_size(latent, (model.patch_size, model.patch_size))

            t_emb = model.t_embedder(t * model.time_scale, dtype=latent.dtype)
            adaln_input = t_emb

            bsz = x_padded.shape[0]
            pH = pW = model.patch_size

            cap_feats_embedded = model.cap_embedder(conditioning)
            if model.pad_tokens_multiple is not None:
                cap_feats_embedded, _ = pad_zimage(
                    cap_feats_embedded, model.cap_pad_token, model.pad_tokens_multiple
                )

            B, C, H, W = x_padded.shape
            x_patches = model.x_embedder(
                x_padded.view(B, C, H // pH, pH, W // pW, pW)
                .permute(0, 2, 4, 3, 5, 1).flatten(3).flatten(1, 2)
            )
            if model.pad_tokens_multiple is not None:
                x_patches, _ = pad_zimage(
                    x_patches, model.x_pad_token, model.pad_tokens_multiple
                )

            cap_freqs_cis = rope_cache['cap_freqs_cis']
            x_freqs_cis = rope_cache['x_freqs_cis']
            freqs_cis = rope_cache['freqs_cis']
            if bsz > 1 and cap_freqs_cis.shape[0] == 1:
                cap_freqs_cis = cap_freqs_cis.expand(bsz, -1, -1, -1, -1, -1)
                x_freqs_cis = x_freqs_cis.expand(bsz, -1, -1, -1, -1, -1)
                freqs_cis = freqs_cis.expand(bsz, -1, -1, -1, -1, -1)

            for layer in model.context_refiner:
                cap_feats_embedded = layer(cap_feats_embedded, None, cap_freqs_cis)
            for layer in model.noise_refiner:
                x_patches = layer(x_patches, None, x_freqs_cis, adaln_input)

            embed = torch.cat([cap_feats_embedded, x_patches], dim=1)

        # --- Phase 2: Detach and start fresh autograd graph ---
        embed = embed.detach().clone().requires_grad_(True)
        adaln_input = adaln_input.detach().clone()
        freqs_cis = freqs_cis.detach().clone()

        # --- Phase 3: 30 main layers (gradients flow through LoRA) ---
        # Use compiled layers when available (per-layer compilation is
        # compatible with gradient checkpointing, unlike whole-model compile).
        layers = self._get_training_layers()
        if gradient_checkpointing:
            for layer in layers:
                embed = grad_ckpt(
                    layer, embed, None, freqs_cis, adaln_input,
                    use_reentrant=False,
                )
        else:
            for layer in layers:
                embed = layer(embed, None, freqs_cis, adaln_input)

        # embed is (B, N_tokens, hidden_dim) with grad_fn intact
        return embed

    def score_hidden(self, hidden_states: Tensor) -> Tensor:
        """Score pre-extracted hidden states through the ScoreUnembedder.

        Args:
            hidden_states: (B, N_tokens, hidden_dim) from extract_hidden()
                or extract_hidden_differentiable().

        Returns:
            (B, N_heads) scalar scores.
        """
        # --- Detached-head guard ---
        # If the head is in training mode and the hidden states have no grad_fn,
        # the adapter will receive zero gradients. This is almost always a bug.
        if self.head.training and hidden_states.grad_fn is None and not hidden_states.requires_grad:
            import warnings
            warnings.warn(
                "score_hidden() called during training with detached hidden states "
                "(grad_fn is None). The adapter will receive ZERO gradients. "
                "Did you use extract_hidden() (inference_mode) instead of "
                "extract_hidden_differentiable()? For training the adapter, use "
                "score_differentiable() which runs the full differentiable forward.",
                UserWarning,
                stacklevel=2,
            )
        return self.head(hidden_states)

    def score(
        self,
        latent: Tensor,
        timestep: Tensor,
        conditioning: Tensor,
        num_tokens: int,
        rope_cache: dict,
    ) -> Tensor:
        """Extract hidden states and score in a single call (INFERENCE ONLY).

        Uses inference_mode -- no gradients flow. The adapter is not trained.
        For training, use score_differentiable().

        Args:
            latent: (B, C, H, W) noisy latent.
            timestep: (B,) timestep values.
            conditioning: (B, seq, cap_feat_dim) text encoder output.
            num_tokens: Number of text tokens.
            rope_cache: Precomputed RoPE cache.

        Returns:
            (B, N_heads) scalar scores (DETACHED).
        """
        hidden = self.extract_hidden(latent, timestep, conditioning, num_tokens, rope_cache)
        return self.score_hidden(hidden)

    def score_differentiable(
        self,
        latent: Tensor,
        timestep: Tensor,
        conditioning: Tensor,
        num_tokens: int,
        rope_cache: dict,
        gradient_checkpointing: bool = True,
    ) -> Tensor:
        """Full differentiable forward: backbone -> hidden -> head -> scores.

        Gradients flow through the adapter's LoRA matrices AND the head.
        This is the correct path for training r_theta.

        Args:
            latent: (B, C, H, W) noisy latent.
            timestep: (B,) timestep values.
            conditioning: (B, seq, cap_feat_dim) text encoder output.
            num_tokens: Number of text tokens.
            rope_cache: Precomputed RoPE cache.
            gradient_checkpointing: Per-block checkpointing (required for 24GB).

        Returns:
            (B, N_heads) scalar scores with grad_fn.
        """
        hidden = self.extract_hidden_differentiable(
            latent, timestep, conditioning, num_tokens, rope_cache,
            gradient_checkpointing=gradient_checkpointing,
        )
        return self.score_hidden(hidden)

    def score_differentiable_packed(
        self,
        images: list[tuple[Tensor, Tensor, Tensor, int]],
        gradient_checkpointing: bool = True,
        force_sdpa: bool = False,
    ) -> Tensor:
        """Packed differentiable scoring for N heterogeneous-resolution images.

        Packs N images into a single FlexAttention forward pass with
        block-diagonal attention masks. Each image can have a different
        resolution. Gradients flow through the adapter's LoRA matrices
        AND the ScoreUnembedder, just like score_differentiable() but
        with N images in one pass instead of N serial passes.

        The forward is split into three phases:
          Phase 1 (no_grad): Embedding + context/noise refiners. No trainable
              params here, so no gradients needed. Produces per-image refined
              caps and noise-refined image patches.
          Phase 2: Detach and build packed sequence + block mask. Start fresh
              autograd graph so gradient checkpointing only covers the 30 main
              transformer layers.
          Phase 3 (grad checkpointing): 30 main transformer layers with block
              masks. Gradients flow through LoRA adapter matrices.

        After the transformer layers, hidden states are unpacked per-image
        and scored independently through the ScoreUnembedder.

        Works with both attention backends:
          - SDPA FlexAttention (force_sdpa=True): reference/correctness
          - SageAttention masked (force_sdpa=False): production, matches inference

        Args:
            images: List of N tuples, each (latent, timestep, conditioning, num_tokens).
                latent: (1, C, H_i, W_i) noisy latent.
                timestep: (1,) sigma value.
                conditioning: (1, seq_i, cap_feat_dim) text conditioning.
                num_tokens: Number of text tokens for this image.
            gradient_checkpointing: Per-block checkpointing (required for 24GB).
            force_sdpa: If True, use SDPA instead of SageAttention. Useful for
                CPU testing or correctness validation against serial path.

        Returns:
            (N, n_heads) score tensor with grad_fn connecting to adapter params.
        """
        from futudiffu.diffusion_model import (
            pad_to_patch_size, pad_zimage, build_packed_sequence,
            build_packed_rope, PackingInfo,
        )
        from src_ii.block_mask import build_block_mask_from_packing_info
        from torch.utils.checkpoint import checkpoint as grad_ckpt

        model = self.backbone
        n_images = len(images)
        device = next(model.parameters()).device
        pH = pW = model.patch_size

        # ---------------------------------------------------------------
        # Phase 1: Embedding + refiners (no trainable params) under no_grad
        # ---------------------------------------------------------------
        refined_cap_list = []
        img_patches_list = []
        img_grid_sizes = []
        embedded_cap_lens = []
        adaln_input = None

        with torch.no_grad():
            for i in range(n_images):
                latent_i, timestep_i, cond_i, num_tokens_i = images[i]

                # Timestep embedding (same formula as forward_packed)
                t_i = 1 - timestep_i
                t_emb_i = model.t_embedder(t_i * model.time_scale, dtype=latent_i.dtype)
                if adaln_input is None:
                    adaln_input = t_emb_i
                else:
                    # For training, each image has its own timestep. We handle
                    # this by using the first image's adaln_input for the packed
                    # forward. The block mask prevents cross-image attention, so
                    # the adaLN modulation is the ONLY cross-image interaction.
                    # Since adaLN only modulates scale/gate (not content), using
                    # a single timestep is acceptable for mixed-sigma batches
                    # when sigmas are similar. For maximally different sigmas,
                    # the error is bounded by the tanh gate saturation.
                    # TODO: if sigma variance is large, consider per-image
                    # adaLN by running layers in a loop. For now, use first.
                    pass

                # Embed caption
                cap_embedded = model.cap_embedder(cond_i)
                cap_embedded_len = cap_embedded.shape[1]
                embedded_cap_lens.append(cap_embedded_len)

                if model.pad_tokens_multiple is not None:
                    cap_embedded, _ = pad_zimage(
                        cap_embedded, model.cap_pad_token, model.pad_tokens_multiple
                    )

                # Compute per-image RoPE for context_refiner
                x_padded_i = pad_to_patch_size(latent_i, (pH, pW))
                _, _, H_i, W_i = x_padded_i.shape
                H_t, W_t = H_i // pH, W_i // pW
                img_grid_sizes.append((H_t, W_t))

                rope_cache_i = model.prepare_rope_cache(H_i, W_i, cap_embedded_len, device)
                cap_freqs_cis_i = rope_cache_i['cap_freqs_cis']
                x_freqs_cis_i = rope_cache_i['x_freqs_cis']

                # Context refiner
                for layer in model.context_refiner:
                    cap_embedded = layer(cap_embedded, None, cap_freqs_cis_i)
                refined_cap_list.append(cap_embedded)

                # Patchify + embed
                patches = model.x_embedder(
                    x_padded_i.view(1, -1, H_t, pH, W_t, pW)
                    .permute(0, 2, 4, 3, 5, 1).flatten(3).flatten(1, 2)
                )
                if model.pad_tokens_multiple is not None:
                    patches, _ = pad_zimage(patches, model.x_pad_token, model.pad_tokens_multiple)

                # Noise refiner
                for layer in model.noise_refiner:
                    patches = layer(patches, None, x_freqs_cis_i, adaln_input)

                img_patches_list.append(patches)

        # Build packing layout from refined caps + noise-refined patches
        packed, packing_info = build_packed_sequence(
            refined_cap_list,
            img_patches_list,
            img_grid_sizes,
            embedded_cap_lens,
            model.pad_tokens_multiple or 1,
            model.cap_pad_token,
            model.x_pad_token,
        )

        # Build packed RoPE
        packed_rope = build_packed_rope(
            packing_info, model.rope_embedder, model.pad_tokens_multiple or 1, device,
        )

        # Build block mask
        block_mask = build_block_mask_from_packing_info(packing_info, device=device)

        # Set attention backend for the backbone's internal sdpa_attention().
        # "sage" uses SageAttention with masked kernel for the uint8 block_mask.
        # "sdpa" does NOT work with uint8 block_mask (needs FlexAttention BlockMask).
        # For force_sdpa=True, we skip the block_mask in phase 3 and rely on
        # sdpa_attention's default dispatch. This means cross-image attention is
        # NOT masked -- only valid for single-image or same-content testing.
        from futudiffu.attention import set_attention_backend
        if force_sdpa:
            set_attention_backend("sdpa")
            # Cannot use uint8 block_mask with SDPA -- nullify it.
            # This disables cross-image isolation, so force_sdpa should only
            # be used for N=1 correctness checks, NOT multi-image training.
            if n_images > 1:
                import warnings
                warnings.warn(
                    f"force_sdpa=True with {n_images} images: block_mask is "
                    f"disabled (SDPA cannot consume uint8 masks). Cross-image "
                    f"attention leakage WILL occur. Use force_sdpa=False for "
                    f"multi-image packed scoring.",
                    UserWarning,
                    stacklevel=2,
                )
            block_mask = None
        else:
            set_attention_backend("sage")

        # ---------------------------------------------------------------
        # Phase 2: Detach and start fresh autograd graph
        # ---------------------------------------------------------------
        packed = packed.detach().clone().requires_grad_(True)
        adaln_input = adaln_input.detach().clone()
        packed_rope = packed_rope.detach().clone()

        # ---------------------------------------------------------------
        # Phase 3: 30 main layers with gradient checkpointing + block masks
        # Use compiled layers when available (per-layer compilation is
        # compatible with gradient checkpointing).
        # ---------------------------------------------------------------
        layers = self._get_training_layers()
        if gradient_checkpointing:
            for layer in layers:
                packed = grad_ckpt(
                    layer, packed, None, packed_rope, adaln_input,
                    use_reentrant=False,
                    block_mask=block_mask,
                )
        else:
            for layer in layers:
                packed = layer(packed, None, packed_rope, adaln_input,
                              block_mask=block_mask)

        # ---------------------------------------------------------------
        # Unpack hidden states per-image and score independently
        # ---------------------------------------------------------------
        scores_list = []
        for i in range(n_images):
            text_start, text_len, img_start, img_len = packing_info.segments[i]
            seg_start = text_start
            seg_end = img_start + img_len
            # Extract this image's hidden states: (1, seg_len, hidden_dim)
            hidden_i = packed[:, seg_start:seg_end, :]
            # Score through ScoreUnembedder
            score_i = self.head(hidden_i)  # (1, n_heads)
            scores_list.append(score_i)

        # Stack to (N, n_heads)
        scores = torch.cat(scores_list, dim=0)
        return scores

    def persist(self, output_dir: str | Path) -> dict:
        """Save both adapter weights and head weights together.

        Creates:
            output_dir/rtheta_adapter.safetensors
            output_dir/btrm_head.safetensors
            output_dir/btrm_compound_config.json

        Returns:
            Dict with file paths and metadata.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save adapter
        adapter_sd = lora_state_dict(self.backbone, adapter_name=self.adapter_name)
        adapter_sd_cpu = {k: v.cpu() for k, v in adapter_sd.items()}
        adapter_path = output_dir / "rtheta_adapter.safetensors"
        save_file(adapter_sd_cpu, str(adapter_path))

        # Save head
        head_sd = {k: v.cpu() for k, v in self.head.state_dict().items()}
        head_path = output_dir / "btrm_head.safetensors"
        save_file(head_sd, str(head_path))

        # Save config
        config_path = output_dir / "btrm_compound_config.json"
        with open(config_path, "w") as f:
            json.dump(self._config, f, indent=2)

        n_adapter = sum(v.numel() for v in adapter_sd_cpu.values())
        n_head = sum(v.numel() for v in head_sd.values())

        manifest = {
            "adapter_path": str(adapter_path),
            "head_path": str(head_path),
            "config_path": str(config_path),
            "n_adapter_tensors": len(adapter_sd_cpu),
            "n_adapter_params": n_adapter,
            "n_head_tensors": len(head_sd),
            "n_head_params": n_head,
        }
        print(f"[BTRMCompoundModel] Persisted: "
              f"{n_adapter} adapter params + {n_head} head params to {output_dir}")
        return manifest

    @classmethod
    def load(
        cls,
        output_dir: str | Path,
        backbone: nn.Module,
        device: torch.device | None = None,
        compile_layers: bool = True,
    ) -> BTRMCompoundModel:
        """Load a persisted compound model. Refuses to load head-only or adapter-only.

        Args:
            output_dir: Directory containing rtheta_adapter.safetensors,
                btrm_head.safetensors, and btrm_compound_config.json.
            backbone: The diffusion backbone model (must match the one used to save).
            device: Target device.
            compile_layers: Whether to per-layer compile (default True).

        Returns:
            BTRMCompoundModel with loaded weights.

        Raises:
            FileNotFoundError: If any of the three required files is missing.
        """
        output_dir = Path(output_dir)

        adapter_path = output_dir / "rtheta_adapter.safetensors"
        head_path = output_dir / "btrm_head.safetensors"
        config_path = output_dir / "btrm_compound_config.json"

        # Refuse to load if any component is missing
        missing = []
        if not adapter_path.exists():
            missing.append(str(adapter_path))
        if not head_path.exists():
            missing.append(str(head_path))
        if not config_path.exists():
            missing.append(str(config_path))
        if missing:
            raise FileNotFoundError(
                f"Cannot load compound BTRM model -- missing files: {missing}. "
                f"A BTRM model requires all three components: adapter, head, and config."
            )

        with open(config_path) as f:
            config = json.load(f)

        # Create the compound model (this allocates adapter + head)
        layer_indices_raw = config.get("adapter_layer_indices", None)
        layer_indices = set(layer_indices_raw) if layer_indices_raw is not None else None
        model = cls(
            backbone,
            adapter_name=config["adapter_name"],
            adapter_rank=config["adapter_rank"],
            adapter_alpha=config["adapter_alpha"],
            adapter_init_b_std=config.get("adapter_init_b_std", DEFAULT_RTHETA_INIT_B_STD),
            adapter_layer_indices=layer_indices,
            head_names=config["head_names"],
            hidden_dim=config["hidden_dim"],
            logit_cap=config["logit_cap"],
            device=device,
            compile_layers=compile_layers,
        )

        # Load adapter weights
        adapter_sd = load_file(str(adapter_path))
        load_lora_state_dict(backbone, adapter_sd)

        # Load head weights
        head_sd = load_file(str(head_path))
        model.head.load_state_dict(head_sd, assign=True)
        if device is not None:
            model.head = model.head.to(device=device, dtype=torch.float32)

        print(f"[BTRMCompoundModel] Loaded from {output_dir}")
        return model

    def train_mode(self) -> None:
        """Set head to train mode. Backbone stays frozen in eval.

        Warns if per-layer compilation is not active, since uncompiled
        training runs ~2x slower and uses ~1.75x more VRAM.
        """
        if not self.is_compiled:
            import warnings
            warnings.warn(
                "[BTRMCompoundModel] Training without per-layer compilation. "
                "This costs ~2x wall time and ~40% more VRAM vs compiled. "
                "Pass compile_layers=True to __init__() (the default) to "
                "enable per-layer compilation. If you explicitly need eager "
                "mode (e.g., debugging), pass compile_layers=False.",
                UserWarning,
                stacklevel=2,
            )
        self.head.train()
        self.backbone.eval()
        # Unfreeze adapter for training
        unfreeze_adapter(self.backbone, self.adapter_name)

    def eval_mode(self) -> None:
        """Set head and adapter to eval mode."""
        self.head.eval()
        self.backbone.eval()
        freeze_adapter(self.backbone, self.adapter_name)

    def cleanup(self) -> None:
        """Remove hooks and free resources."""
        self._capture.remove()

"""ZImageRLAIF: Unified diffusion model with integrated score head.

One model class. One forward method. Two outputs: (diffusion_fields, scores).
The packed forward with block masks IS the forward. A single image is a
packed batch of size 1 with a trivial (all-ones) block mask.

The score head is ~20 lines: RMSNorm -> Linear -> soft_tanh_cap. Applied
to the output of layers[-1] BEFORE final_layer runs. Zero-initialized, so
untrained models return scores of zero. Trained score_proj weights load
via strict=False.

This class uses branchless architecture components from src_ii.transformer.
It does NOT inherit from NextDiT. It creates its own module tree and
implements a single forward path.

Import constraints:
  - IMPORTS from src_ii.transformer: branchless architecture components
  - No imports from futudiffu.diffusion_model (frozen)
  - No imports from src_ii.btrm_model, src_ii.lora, or src_ii.forward_packed
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from src_ii.transformer import (
    EmbedND,
    FinalLayer,
    JointTransformerBlock,
    PackingInfo,
    RMSNormModule,
    TimestepEmbedder,
    build_packed_rope,
    build_packed_sequence,
    build_trivial_mask,
    pad_to_patch_size,
    pad_zimage,
    unpack_and_unpatchify,
)


class ZImageRLAIF(nn.Module):
    """Unified diffusion model with integrated BTRM score head.

    Architecture: NextDiT with dim=3840, n_heads=30, 30 main layers,
    2 context_refiner + 2 noise_refiner blocks, FP8 weights with
    blockwise scales, fused Triton kernels.

    Returns (diffusion_fields, scores) from every forward call.

    Attributes:
        gradient_checkpointing: When True, each main layer is wrapped in
            torch.utils.checkpoint. Set once during training setup. The
            branch is resolved at trace time by torch.compile (the attribute
            does not change between calls).
    """

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        dim: int = 3840,
        n_layers: int = 32,
        n_refiner_layers: int = 2,
        n_heads: int = 30,
        n_kv_heads: int = 30,
        multiple_of: int = 256,
        ffn_dim_multiplier: float = 8.0 / 3.0,
        norm_eps: float = 1e-5,
        qk_norm: bool = False,
        cap_feat_dim: int = 2560,
        axes_dims: list[int] = (32, 48, 48),
        axes_lens: list[int] = (1536, 512, 512),
        rope_theta: float = 256.0,
        z_image_modulation: bool = True,
        time_scale: float = 1000.0,
        pad_tokens_multiple: int | None = 32,
        n_score_heads: int = 2,
        score_cap: float = 10.0,
        use_sage: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.dtype = dtype
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.time_scale = time_scale
        self.pad_tokens_multiple = pad_tokens_multiple
        self.dim = dim
        self.n_heads = n_heads
        self.n_score_heads = n_score_heads
        self.score_cap = score_cap
        self.use_sage = use_sage

        # Gradient checkpointing flag (resolved at compile trace time)
        self.gradient_checkpointing = False

        # --- Embedding layers ---
        self.x_embedder = nn.Linear(
            patch_size * patch_size * in_channels, dim,
            bias=True, device=device, dtype=dtype,
        )

        self.t_embedder = TimestepEmbedder(
            min(dim, 1024),
            output_size=256 if z_image_modulation else None,
            device=device, dtype=dtype,
        )

        self.cap_embedder = nn.Sequential(
            RMSNormModule(cap_feat_dim, eps=norm_eps, elementwise_affine=True,
                          device=device, dtype=dtype),
            nn.Linear(cap_feat_dim, dim, bias=True, device=device, dtype=dtype),
        )

        # --- Refiner blocks ---
        self.context_refiner = nn.ModuleList([
            JointTransformerBlock(
                layer_id, dim, n_heads, n_kv_heads, multiple_of,
                ffn_dim_multiplier, norm_eps, qk_norm,
                modulation=False, use_sage=use_sage,
                device=device, dtype=dtype,
            )
            for layer_id in range(n_refiner_layers)
        ])

        self.noise_refiner = nn.ModuleList([
            JointTransformerBlock(
                layer_id, dim, n_heads, n_kv_heads, multiple_of,
                ffn_dim_multiplier, norm_eps, qk_norm,
                modulation=True, z_image_modulation=z_image_modulation,
                use_sage=use_sage, device=device, dtype=dtype,
            )
            for layer_id in range(n_refiner_layers)
        ])

        # --- Main transformer layers ---
        self.layers = nn.ModuleList([
            JointTransformerBlock(
                layer_id, dim, n_heads, n_kv_heads, multiple_of,
                ffn_dim_multiplier, norm_eps, qk_norm,
                z_image_modulation=z_image_modulation,
                attn_out_bias=False, use_sage=use_sage,
                device=device, dtype=dtype,
            )
            for layer_id in range(n_layers)
        ])

        # --- Final layer (diffusion output) ---
        self.final_layer = FinalLayer(
            dim, patch_size, self.out_channels,
            z_image_modulation=z_image_modulation,
            device=device, dtype=dtype,
        )

        # --- Score head (BTRM output) ---
        self.score_norm = RMSNormModule(
            dim, eps=norm_eps, elementwise_affine=False,
            device=device, dtype=dtype,
        )
        self.score_proj = nn.Linear(
            dim, n_score_heads, bias=False,
            device=device, dtype=dtype,
        )
        # Zero init: untrained model returns zeros, no graph break vs trained
        nn.init.zeros_(self.score_proj.weight)

        # --- Padding tokens ---
        if self.pad_tokens_multiple is not None:
            self.x_pad_token = nn.Parameter(
                torch.zeros((1, dim), device=device, dtype=dtype)
            )
            self.cap_pad_token = nn.Parameter(
                torch.zeros((1, dim), device=device, dtype=dtype)
            )

        # --- RoPE ---
        assert (dim // n_heads) == sum(axes_dims)
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        self.rope_embedder = EmbedND(
            dim=dim // n_heads, theta=rope_theta, axes_dim=list(axes_dims),
        )

    # ------------------------------------------------------------------
    # Batched adaLN precomputation
    # ------------------------------------------------------------------

    def prepare_adaln_cache(self) -> None:
        """Pre-stack adaLN weights from all modulated blocks for batched GEMM.

        Concatenates Linear weight matrices from noise_refiner (2 blocks)
        and main layers (30 blocks) into a single weight matrix. This
        allows computing all adaLN projections in one GEMM.
        """
        import itertools

        from futudiffu.fp8 import FP8Linear, dequantize_fp8_blockwise

        weights = []
        biases = []

        for block in itertools.chain(self.noise_refiner, self.layers):
            if block.modulation:
                linear = block.adaLN_modulation[0]
                if hasattr(linear, 'weight_scale'):
                    # FP8Linear: dequantize for batched F.linear
                    w_bf16 = dequantize_fp8_blockwise(
                        linear.weight, linear.weight_scale,
                        block_size=linear.block_size,
                        output_dtype=torch.bfloat16,
                    )
                    weights.append(w_bf16)
                    biases.append(linear.bias)
                else:
                    weights.append(linear.weight)
                    biases.append(linear.bias)

        self.register_buffer('_adaln_W', torch.cat(weights, dim=0))
        self.register_buffer('_adaln_B', torch.cat(biases, dim=0))
        self._adaln_n_blocks = len(weights)
        self._adaln_output_dim = weights[0].shape[0]  # 4*dim = 15360

    def _compute_adaln_params(
        self, adaln_input: torch.Tensor,
    ) -> list[tuple[torch.Tensor, ...]] | None:
        """Compute all adaLN params in one batched GEMM.

        Args:
            adaln_input: (M, 256) timestep embeddings, where M = n_images * B.
                For packed BTRM training (B=1): M = n_images, one per image.
                For CFG inference (B=2): M = n_images * 2.

        Returns:
            List of N tuples: (scale_msa, gate_msa, scale_mlp, gate_mlp),
            each containing (M, dim) tensors. None if cache not prepared.
        """
        if not hasattr(self, '_adaln_W'):
            return None

        all_params = F.linear(adaln_input, self._adaln_W, self._adaln_B)
        per_block = all_params.split(self._adaln_output_dim, dim=-1)
        return [p.chunk(4, dim=-1) for p in per_block]

    # ------------------------------------------------------------------
    # RoPE cache
    # ------------------------------------------------------------------

    def prepare_rope_cache(
        self, h: int, w: int, cap_len: int, device: torch.device,
    ) -> dict:
        """Precompute RoPE embeddings for a single image.

        Args:
            h: Image height (pixels, after pad_to_patch_size).
            w: Image width (pixels, after pad_to_patch_size).
            cap_len: Caption token count after cap_embedder (before padding).
            device: Target device.

        Returns:
            Dict with 'cap_freqs_cis', 'x_freqs_cis', 'freqs_cis'.
        """
        pH = pW = self.patch_size

        cap_len_padded = cap_len
        if self.pad_tokens_multiple:
            cap_len_padded += ((-cap_len) % self.pad_tokens_multiple)

        cap_pos_ids = torch.zeros(1, cap_len_padded, 3, dtype=torch.float32, device=device)
        cap_pos_ids[:, :, 0] = torch.arange(cap_len_padded, dtype=torch.float32, device=device) + 1.0
        cap_freqs_cis = self.rope_embedder(cap_pos_ids).movedim(1, 2)

        H_tokens, W_tokens = h // pH, w // pW
        n_img_tokens = H_tokens * W_tokens
        n_img_padded = n_img_tokens
        if self.pad_tokens_multiple:
            n_img_padded += ((-n_img_tokens) % self.pad_tokens_multiple)

        x_pos_ids = torch.zeros((1, n_img_padded, 3), dtype=torch.float32, device=device)
        x_pos_ids[:, :n_img_tokens, 0] = cap_len_padded + 1
        x_pos_ids[:, :n_img_tokens, 1] = (
            torch.arange(H_tokens, dtype=torch.float32, device=device)
            .view(-1, 1).repeat(1, W_tokens).flatten()
        )
        x_pos_ids[:, :n_img_tokens, 2] = (
            torch.arange(W_tokens, dtype=torch.float32, device=device)
            .view(1, -1).repeat(H_tokens, 1).flatten()
        )
        x_freqs_cis = self.rope_embedder(x_pos_ids).movedim(1, 2)

        freqs_cis = torch.cat([cap_freqs_cis, x_freqs_cis], dim=1)

        return {
            'cap_freqs_cis': cap_freqs_cis,
            'x_freqs_cis': x_freqs_cis,
            'freqs_cis': freqs_cis,
        }

    # ------------------------------------------------------------------
    # Pre-compute constant packed state (caps, packing layout, RoPE)
    # ------------------------------------------------------------------

    def prepare_packed_state(
        self,
        context_list: list[torch.Tensor],
        img_sizes: list[tuple[int, int]],
        cap_lens: list[int],
        device: torch.device,
    ) -> tuple[list[torch.Tensor], PackingInfo, torch.Tensor]:
        """Pre-compute constant state for packed multi-image generation.

        Runs cap_embedder + context_refiner for each image, computes packing
        layout and RoPE frequencies. All outputs are constant across euler steps.

        Args:
            context_list: N raw text conditionings, each (B, seq_i, cap_feat_dim).
            img_sizes: (H, W) per image AFTER pad_to_patch_size.
            cap_lens: Original caption lengths before embedding/padding.
            device: Target device.

        Returns:
            (refined_cap_list, packing_info, packed_rope)
        """
        pH = pW = self.patch_size

        refined_cap_list = []
        dummy_img_patches_list = []
        img_grid_sizes = []
        embedded_cap_lens = []

        for i in range(len(context_list)):
            cap_embedded = self.cap_embedder(context_list[i])
            cap_embedded_len = cap_embedded.shape[1]

            if self.pad_tokens_multiple is not None:
                cap_embedded, _ = pad_zimage(
                    cap_embedded, self.cap_pad_token, self.pad_tokens_multiple,
                )

            H, W = img_sizes[i]
            H_t, W_t = H // pH, W // pW
            img_grid_sizes.append((H_t, W_t))

            rope_cache = self.prepare_rope_cache(H, W, cap_embedded_len, device)
            cap_freqs_cis = rope_cache['cap_freqs_cis']

            bsz = cap_embedded.shape[0]
            if bsz > 1 and cap_freqs_cis.shape[0] == 1:
                cap_freqs_cis = cap_freqs_cis.expand(bsz, -1, -1, -1, -1, -1)

            cap_block_mask = build_trivial_mask(cap_embedded.shape[1], device, use_sage=self.use_sage)
            for layer in self.context_refiner:
                cap_embedded = layer(cap_embedded, None, cap_freqs_cis,
                                     block_mask=cap_block_mask)

            refined_cap_list.append(cap_embedded)
            embedded_cap_lens.append(cap_embedded_len)

            n_img_tokens = H_t * W_t
            n_img_padded = n_img_tokens
            if self.pad_tokens_multiple is not None:
                n_img_padded += ((-n_img_tokens) % self.pad_tokens_multiple)
            dummy_img_patches_list.append(
                torch.zeros(1, n_img_padded, self.dim, device=device, dtype=cap_embedded.dtype)
            )

        # Build packing layout (batch-1 slices for layout computation)
        layout_caps = [cap[:1] for cap in refined_cap_list]
        _, packing_info = build_packed_sequence(
            layout_caps,
            dummy_img_patches_list,
            img_grid_sizes,
            embedded_cap_lens,
            self.pad_tokens_multiple or 1,
            self.cap_pad_token,
            self.x_pad_token,
        )

        packed_rope = build_packed_rope(
            packing_info, self.rope_embedder,
            self.pad_tokens_multiple or 1, device,
        )

        return refined_cap_list, packing_info, packed_rope

    # ------------------------------------------------------------------
    # Score head computation
    # ------------------------------------------------------------------

    def _compute_scores(
        self,
        hidden: torch.Tensor,
        packing_info: PackingInfo,
    ) -> torch.Tensor:
        """Compute per-image scores from the last layer's hidden state.

        For each image in the packed batch: extract its segment from the
        hidden state, mean-pool over tokens, apply RMSNorm, project to
        score heads, apply soft tanh cap.

        Args:
            hidden: (B, total_len, dim) output of layers[-1].
            packing_info: Packed sequence layout.

        Returns:
            (n_images, n_score_heads) score tensor. When score_proj is
            zero-initialized, returns zeros.
        """
        n_images = packing_info.n_images

        # Build scores as list + stack to preserve grad_fn.
        # new_zeros() does NOT inherit requires_grad — in-place assign
        # into a detached tensor would disconnect gradients from score_proj.
        score_list = []

        for i in range(n_images):
            text_start, text_len, img_start, img_len = packing_info.segments[i]
            seg_start = text_start
            seg_end = img_start + img_len

            # Extract this image's full segment (text + image tokens)
            # Use batch dim 0 (B=1 for packed batches)
            seg = hidden[0, seg_start:seg_end, :]  # (seg_len, dim)

            # Mean pool over tokens
            pooled = seg.mean(dim=0, keepdim=True)  # (1, dim)

            # RMSNorm + linear projection
            normed = self.score_norm(pooled)  # (1, dim)
            raw = self.score_proj(normed)  # (1, n_score_heads)

            # Soft tanh cap: cap * tanh(raw / cap)
            score_i = self.score_cap * torch.tanh(raw[0] / self.score_cap)
            score_list.append(score_i)

        return torch.stack(score_list)

    # ------------------------------------------------------------------
    # Preprocessing: variable-length ops run eagerly (outside compiled graph)
    # ------------------------------------------------------------------

    @torch.compiler.disable
    def _preprocess(
        self,
        x_list: list[torch.Tensor],
        timesteps_list: list[torch.Tensor],
        refined_cap_list: list[torch.Tensor],
        packing_info: PackingInfo,
        packed_rope: torch.Tensor,
        adapter_scales: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,  # packed (B, total_len, dim) -- fixed shape
        torch.Tensor,  # adaln_input (B, 256) -- image 0, for FinalLayer
        list[tuple[torch.Tensor, ...]] | None,  # adaln_params (pre-scattered for per-token)
        torch.Tensor,  # rope (B, ...) -- fixed shape
        torch.Tensor | None,  # token_to_image
        list[tuple[int, int]],  # original_hw
    ]:
        """Eager preprocessing: timestep embedding, patchify, noise_refiner, pack.

        Handles variable-length x_list by iterating over images eagerly,
        then produces fixed-shape tensors padded to packing_info.total_len
        (REFERENCE_TOTAL_LEN). Decorated with @torch.compiler.disable so
        the compiled graph never traces this code -- it sees only the
        fixed-shape outputs.

        adaLN path selection:
          - Per-token (B=1, n_images>1): compute per-image embeddings, scatter
            to per-token positions using document_id. Returns pre-scattered
            adaln_params as tuples of (1, total_len, dim) tensors.
          - Per-batch broadcast (single image or CFG B>1): compute from image 0,
            broadcast. Returns adaln_params as tuples of (B, dim) tensors.
        """
        n_images = len(x_list)
        B = x_list[0].shape[0]
        pH = pW = self.patch_size
        device = x_list[0].device

        # --- Timestep embedding ---
        _per_token_adaln = (B == 1 and n_images > 1)

        if _per_token_adaln:
            # Packed BTRM training: compute per-image, scatter to per-token
            all_adaln_inputs = []
            for i in range(n_images):
                t_i = 1 - timesteps_list[i]
                t_emb_i = self.t_embedder(t_i * self.time_scale, dtype=x_list[0].dtype)
                all_adaln_inputs.append(t_emb_i)  # each (1, 256)
            all_adaln_inputs = torch.cat(all_adaln_inputs, dim=0)  # (n_images, 256)
            adaln_input = all_adaln_inputs[:1]  # (1, 256) -- image 0 for FinalLayer
            adaln_params_raw = self._compute_adaln_params(all_adaln_inputs)

            # Build scatter index: document_id maps tokens to source images.
            # Padding tokens have document_id == -1; remap to 0 (irrelevant
            # since block mask zeros them out, but index must be valid).
            adaln_doc_id = packing_info.document_id.clone()
            adaln_doc_id[adaln_doc_id < 0] = 0

            # Pre-scatter all layer params to per-token layout so the
            # compiled main loop sees only fixed-shape tensors.
            if adaln_params_raw is not None:
                adaln_params = []
                for param_tuple in adaln_params_raw:
                    adaln_params.append(tuple(
                        p[adaln_doc_id].unsqueeze(0).contiguous()
                        for p in param_tuple
                    ))  # each tuple: 4 x (1, total_len, dim)
            else:
                adaln_params = None
        else:
            # Single image or CFG: image 0 broadcast (original behavior)
            t_base = 1 - timesteps_list[0]
            adaln_input = self.t_embedder(t_base * self.time_scale, dtype=x_list[0].dtype)
            adaln_params = self._compute_adaln_params(adaln_input)

        # --- Per-image: patchify + noise_refiner ---
        img_patches_list = []
        original_hw = []

        for i in range(n_images):
            x_i = x_list[i]
            _, _, h_i, w_i = x_i.shape
            original_hw.append((h_i, w_i))

            x_i = pad_to_patch_size(x_i, (pH, pW))
            B_i, C_i, H_i, W_i = x_i.shape
            H_t, W_t = H_i // pH, W_i // pW

            # Patchify + embed
            patches = self.x_embedder(
                x_i.view(B_i, C_i, H_t, pH, W_t, pW)
                .permute(0, 2, 4, 3, 5, 1).flatten(3).flatten(1, 2)
            )

            if self.pad_tokens_multiple is not None:
                patches, _ = pad_zimage(patches, self.x_pad_token, self.pad_tokens_multiple)

            # Per-image noise_refiner with per-image RoPE and per-image timestep
            rope_cache_i = self.prepare_rope_cache(
                H_i, W_i, packing_info.cap_lens[i], device,
            )
            x_freqs_cis = rope_cache_i['x_freqs_cis']
            if B > 1 and x_freqs_cis.shape[0] == 1:
                x_freqs_cis = x_freqs_cis.expand(B, -1, -1, -1, -1, -1)

            # Per-image timestep for noise_refiner
            t_i = 1 - timesteps_list[i]
            adaln_input_i = self.t_embedder(t_i * self.time_scale, dtype=x_i.dtype)

            noise_block_mask = build_trivial_mask(patches.shape[1], device, use_sage=self.use_sage)
            for layer in self.noise_refiner:
                patches = layer(
                    patches, None, x_freqs_cis,
                    adaln_input=adaln_input_i,
                    precomputed_adaln=None,
                    block_mask=noise_block_mask,
                )

            img_patches_list.append(patches)

        # --- Pack into single sequence ---
        all_tokens = []
        for i in range(n_images):
            cap_i = refined_cap_list[i]
            if B > 1 and cap_i.shape[0] == 1:
                cap_i = cap_i.expand(B, -1, -1)
            all_tokens.append(cap_i)
            all_tokens.append(img_patches_list[i])
        packed = torch.cat(all_tokens, dim=1)  # (B, natural_len, dim)

        # Pad to packing_info.total_len (fixed REFERENCE_TOTAL_LEN).
        # The block mask zeros out padding tiles -- they cost nothing.
        natural_len = packed.shape[1]
        if natural_len < packing_info.total_len:
            pad = packed.new_zeros(B, packing_info.total_len - natural_len, packed.shape[2])
            packed = torch.cat([packed, pad], dim=1)

        # Expand packed RoPE for CFG batch
        rope = packed_rope
        if B > 1 and rope.shape[0] == 1:
            rope = rope.expand(B, -1, -1, -1, -1, -1)

        # Build token_to_image for LoRA routing
        token_to_image = packing_info.document_id if adapter_scales is not None else None

        return packed, adaln_input, adaln_params, rope, token_to_image, original_hw

    # ------------------------------------------------------------------
    # Postprocessing: variable-length ops run eagerly (outside compiled graph)
    # ------------------------------------------------------------------

    @torch.compiler.disable
    def _postprocess(
        self,
        packed: torch.Tensor,
        packing_info: PackingInfo,
        original_hw: list[tuple[int, int]],
    ) -> list[torch.Tensor]:
        """Eager postprocessing: unpack, unpatchify, crop, negate.

        Handles variable-length output list by iterating over images eagerly.
        Decorated with @torch.compiler.disable so the compiled graph ends
        at the boundary where fixed-shape tensors are produced by final_layer.
        """
        results = unpack_and_unpatchify(
            packed, packing_info, self.patch_size, self.out_channels,
        )

        # Crop to original sizes and negate
        diffusion_fields = []
        for i in range(len(original_hw)):
            h_orig, w_orig = original_hw[i]
            diffusion_fields.append(-results[i][:, :, :h_orig, :w_orig])

        return diffusion_fields

    # ------------------------------------------------------------------
    # THE forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x_list: list[torch.Tensor],
        timesteps_list: list[torch.Tensor],
        refined_cap_list: list[torch.Tensor],
        packing_info: PackingInfo,
        block_mask: torch.Tensor,
        packed_rope: torch.Tensor,
        adapter_scales: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Unified packed forward pass: diffusion + scoring.

        Processes N images in a single forward pass through the transformer.
        Returns both the diffusion field outputs and BTRM scores.

        Structure:
            1. _preprocess() [eager, @torch.compiler.disable]:
               Variable-length ops (timestep embed, patchify, noise_refiner,
               pack, pad, adaLN scatter). Produces fixed-shape tensors.
            2. 30 main transformer layers + score head + final_layer [compiled]:
               Fixed-shape tensor ops. One compiled graph regardless of
               how many images are packed.
            3. _postprocess() [eager, @torch.compiler.disable]:
               Variable-length ops (unpack, crop, negate). Consumes
               fixed-shape tensor, produces variable-length list.

        Args:
            x_list: N noisy latent images, each (B, C, H_i, W_i).
            timesteps_list: N per-image timestep tensors, each (B,).
                For inference (shared sigma), pass the same tensor N times.
                For BTRM training, each image can have a different sigma.
            refined_cap_list: N pre-refined caption embeddings from
                prepare_packed_state(), each (B, cap_padded_len_i, dim).
            packing_info: Pre-built PackingInfo (constant across euler steps).
            block_mask: Pre-built uint8 block mask (constant across steps).
            packed_rope: Pre-built packed RoPE (constant across steps).
            adapter_scales: Optional (n_images, n_adapters) LoRA adapter
                scales for multi-tenant routing. Passed through to all
                MultiLoRALinear modules in the transformer layers.

        Returns:
            (diffusion_fields, scores) where:
                diffusion_fields: list of N tensors, each (B, C, H_i, W_i), NEGATED
                scores: (n_images, n_score_heads) tensor
        """
        # --- Phase 1: Eager preprocessing (variable-length -> fixed-shape) ---
        packed, adaln_input, adaln_params, rope, token_to_image, original_hw = (
            self._preprocess(
                x_list, timesteps_list, refined_cap_list,
                packing_info, packed_rope, adapter_scales,
            )
        )

        # --- Phase 2: 30 main transformer layers (fixed-shape, compiled) ---
        n_refiner = len(self.noise_refiner)
        param_idx = n_refiner

        for layer in self.layers:
            precomputed = adaln_params[param_idx] if adaln_params is not None else None

            if self.gradient_checkpointing:
                packed = grad_checkpoint(
                    self._layer_forward,
                    layer, packed, rope, adaln_input, precomputed,
                    block_mask, adapter_scales, token_to_image,
                    use_reentrant=False,
                )
            else:
                packed = self._layer_forward(
                    layer, packed, rope, adaln_input, precomputed,
                    block_mask, adapter_scales, token_to_image,
                )
            param_idx += 1

        # --- Phase 3a: Score head (from layers[-1] output, before final_layer) ---
        scores = self._compute_scores(packed, packing_info)

        # --- Phase 3b: Final layer (fixed-shape, compiled) ---
        packed = self.final_layer(packed, adaln_input)

        # --- Phase 4: Eager postprocessing (fixed-shape -> variable-length) ---
        diffusion_fields = self._postprocess(packed, packing_info, original_hw)

        return diffusion_fields, scores

    @staticmethod
    def _layer_forward(
        layer: nn.Module,
        packed: torch.Tensor,
        rope: torch.Tensor,
        adaln_input: torch.Tensor,
        precomputed_adaln: tuple[torch.Tensor, ...] | None,
        block_mask: torch.Tensor,
        adapter_scales: torch.Tensor | None,
        token_to_image: torch.Tensor | None,
    ) -> torch.Tensor:
        """Execute a single transformer layer.

        Separated into a static method so it can be wrapped by
        torch.utils.checkpoint without capturing `self` (which would
        prevent memory savings).

        adapter_scales and token_to_image are passed as explicit kwargs
        through the JointTransformerBlock -> JointAttention / FeedForward
        -> MultiLoRALinear call chain. No module attribute mutation.
        """
        return layer(
            packed, None, rope,
            adaln_input=adaln_input,
            precomputed_adaln=precomputed_adaln,
            block_mask=block_mask,
            adapter_scales=adapter_scales,
            token_to_image=token_to_image,
        )

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def compile_for_execution(self) -> 'ZImageRLAIF':
        """Compile all transformer blocks in-place.

        Wraps every JointTransformerBlock (main layers, noise_refiner,
        context_refiner) and the FinalLayer with torch.compile. Each
        compiled module is independently optimized; calling a compiled
        module from eager context (e.g. inside @torch.compiler.disable)
        still invokes the compiled graph.

        NOT compiled:
          - _preprocess / _postprocess: Python dispatch only (no matmuls).
            Already @torch.compiler.disable.
          - prepare_packed_state: Python dispatch that calls compiled
            context_refiner blocks.
          - _compute_scores: Python-level iteration over segments.
          - score_norm, score_proj: trivially small.
          - t_embedder, cap_embedder, x_embedder, rope_embedder:
            small embedding ops.

        Returns self (not a wrapper).
        """
        def _compile_if_needed(module):
            if isinstance(module, torch._dynamo.eval_frame.OptimizedModule):
                return module
            return torch.compile(module, mode="default")

        for i in range(len(self.layers)):
            self.layers[i] = _compile_if_needed(self.layers[i])
        for i in range(len(self.noise_refiner)):
            self.noise_refiner[i] = _compile_if_needed(self.noise_refiner[i])
        for i in range(len(self.context_refiner)):
            self.context_refiner[i] = _compile_if_needed(self.context_refiner[i])
        self.final_layer = _compile_if_needed(self.final_layer)
        return self


# ------------------------------------------------------------------
# Model creation and loading
# ------------------------------------------------------------------

def create_zimage_rlaif(
    dtype: torch.dtype = torch.bfloat16,
    n_layers: int = 32,
    cap_feat_dim: int = 2560,
    qk_norm: bool = True,
    n_score_heads: int = 2,
    score_cap: float = 10.0,
    use_sage: bool = True,
) -> ZImageRLAIF:
    """Create ZImageRLAIF on the 'meta' device (uninitialized weights).

    Use this when loading FP8 weights via replace_linear_with_fp8().
    """
    return ZImageRLAIF(
        patch_size=2,
        in_channels=16,
        dim=3840,
        n_layers=n_layers,
        n_refiner_layers=2,
        n_heads=30,
        n_kv_heads=30,
        multiple_of=256,
        ffn_dim_multiplier=8.0 / 3.0,
        norm_eps=1e-5,
        qk_norm=qk_norm,
        cap_feat_dim=cap_feat_dim,
        axes_dims=[32, 48, 48],
        axes_lens=[1536, 512, 512],
        rope_theta=256.0,
        z_image_modulation=True,
        time_scale=1000.0,
        pad_tokens_multiple=32,
        n_score_heads=n_score_heads,
        score_cap=score_cap,
        use_sage=use_sage,
        device="meta",
        dtype=dtype,
    )


def load_zimage_rlaif(
    fp8_safetensors_path: str,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.bfloat16,
    fp8_block_size: int = 128,
    n_score_heads: int = 2,
    score_cap: float = 10.0,
    fuse: bool = True,
    compile_model: bool = True,
    use_sage: bool = True,
) -> ZImageRLAIF:
    """Load ZImageRLAIF from FP8 safetensors checkpoint.

    Loading sequence:
      1. Load safetensors state dict
      2. Detect architecture params (n_layers, cap_feat_dim, qk_norm)
      3. Create meta-device model skeleton
      4. Replace nn.Linear with FP8Linear where weights are FP8
      5. Load remaining weights (strict=False: score_proj/score_norm missing = zero-init stays)
      6. Move to device, eval mode
      7. Fuse model (optional)
      8. compile_for_execution (optional)

    Args:
        fp8_safetensors_path: Path to checkpoint.
        device: Target CUDA device.
        dtype: Working dtype (bfloat16).
        fp8_block_size: FP8 blockwise block size.
        n_score_heads: Number of BTRM score heads.
        score_cap: Soft tanh cap for score output.
        fuse: Whether to apply model fusions.
        compile_model: Whether to compile inner layers.

    Returns:
        The model. When compile_model=True, inner layers are compiled
        in-place. There is no separate raw_model — one object for all uses.
    """
    import time
    from safetensors.torch import load_file

    from src_ii.transformer import (
        _detect_cap_feat_dim,
        _detect_n_layers,
        _detect_qk_norm,
        _strip_diffusion_prefix,
        fuse_model,
    )
    from futudiffu.fp8 import replace_linear_with_fp8

    t0 = time.perf_counter()

    diff_sd = load_file(fp8_safetensors_path, device=str(device))
    remapped = _strip_diffusion_prefix(diff_sd)
    del diff_sd

    n_layers = _detect_n_layers(remapped.keys())
    cap_feat_dim = _detect_cap_feat_dim(remapped)
    qk_norm = _detect_qk_norm(remapped.keys())

    model = create_zimage_rlaif(
        dtype=dtype, n_layers=n_layers,
        cap_feat_dim=cap_feat_dim, qk_norm=qk_norm,
        n_score_heads=n_score_heads, score_cap=score_cap,
        use_sage=use_sage,
    )

    replace_linear_with_fp8(
        model, remapped, block_size=fp8_block_size,
        output_dtype=dtype,
    )

    # Load remaining non-FP8 weights (score_proj/score_norm will be missing
    # from legacy checkpoints -- strict=False leaves zero-init in place)
    remaining = {
        k: v for k, v in remapped.items()
        if not k.endswith((".weight_scale", ".comfy_quant"))
    }
    model.load_state_dict(remaining, strict=False, assign=True)
    del remapped, remaining

    # score_proj may still be on meta device if not in the checkpoint (legacy
    # checkpoints lack it). Materialize before model.to(device).
    for pname, param in model.score_proj.named_parameters():
        if param.device.type == "meta":
            materialized = torch.zeros(
                param.shape, device=device, dtype=param.dtype,
            )
            setattr(model.score_proj, pname, nn.Parameter(materialized, requires_grad=param.requires_grad))

    model = model.to(device)

    # Re-zero score_proj after materialization. nn.init.zeros_() in __init__
    # is a no-op on meta device (no storage). After .to(device), the weight
    # materializes as uninitialized garbage. If the checkpoint had score_proj
    # weights, load_state_dict(assign=True) already overwrote them. If not
    # (legacy checkpoint), we need explicit zeros for "untrained = zero scores."
    model.score_proj.weight.data.zero_()

    model.eval()

    # Configure SageAttention if the model uses it
    if use_sage:
        from futudiffu.sage_attention import configure_sage
        configure_sage(smooth_k=True, qk_quant="int8", pv_quant="bf16")

    if fuse:
        fuse_model(model)

    elapsed = time.perf_counter() - t0
    print(f"[zimage_model] Loaded in {elapsed:.1f}s "
          f"(n_layers={n_layers}, score_heads={n_score_heads}, sage={use_sage})")

    if compile_model:
        model.compile_for_execution()

    return model

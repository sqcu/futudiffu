"""Reusable attention capture for mechanistic interpretability.

Extracted from scripts_ii/attention_interpretability.py. Monkey-patches
sdpa_attention to compute per-token attention statistics (per-head streaming
to avoid OOM) while using the efficient SDPA path for actual model output.

Usage:
    capture = AttentionCapture()
    capture.install()
    stats = capture.capture_forward(model, latent, timestep, conditioning, ...)
    capture.remove()

Import constraints:
  - IMPORTS from futudiffu.attention: for monkey-patching sdpa_attention
  - torch for tensor operations
  - DOES NOT import: model_manager, server, client
"""

from __future__ import annotations

import torch
from torch import Tensor


class AttentionCapture:
    """Monkey-patches sdpa_attention to capture per-token attention stats.

    For each layer's attention call, computes:
      - attn_received: (n_heads, seq_len) -- mean attention each token receives
        (mean over query dimension of softmax weights)
      - attn_given:    (n_heads, seq_len) -- mean attention each token gives
        (mean over key dimension of softmax weights)
      - head_norms:    (n_heads,) -- L2 norm of the full attention weight matrix
        per head (approximated as sqrt of sum of squared means)

    Uses per-head streaming to avoid storing full (n_heads, seq_len, seq_len)
    matrices. Peak overhead: ~64 MB at seq=4k (one head at a time),
    vs ~1.9 GB for all 30 heads at once.
    """

    def __init__(self) -> None:
        self.layer_stats: dict[int, dict] = {}
        self._call_count = 0
        self._original_fn = None
        self._active = False

    def install(self) -> None:
        """Monkey-patch futudiffu.attention.sdpa_attention."""
        import futudiffu.attention as attn_mod
        self._original_fn = attn_mod.sdpa_attention
        self._call_count = 0
        self.layer_stats = {}

        capture = self

        def patched_sdpa_attention(
            q: Tensor, k: Tensor, v: Tensor,
            heads: int, mask: Tensor | None = None,
            skip_reshape: bool = False, block_mask=None,
        ) -> Tensor:
            # Ensure we have (B, heads, seq, dim) format
            if skip_reshape:
                b, _, seq_len, dim_head = q.shape
            else:
                b, seq_total, dim_total = q.shape
                dim_head = dim_total // heads
                q = q.view(b, -1, heads, dim_head).transpose(1, 2)
                k = k.view(b, -1, heads, dim_head).transpose(1, 2)
                v = v.view(b, -1, heads, dim_head).transpose(1, 2)
                seq_len = q.shape[2]

            # --- Reduced-memory attention statistics ---
            # NEVER materialise the full (B, heads, seq, seq) attention matrix.
            # Instead accumulate per-query row statistics one head at a time.

            with torch.no_grad():
                scale = 1.0 / (dim_head ** 0.5)

                # Accumulators for reduced stats (float32, on-device)
                attn_received = q.new_zeros(heads, seq_len, dtype=torch.float32)
                attn_given = q.new_zeros(heads, seq_len, dtype=torch.float32)
                sq_accum = q.new_zeros(heads, dtype=torch.float32)

                # Process one head at a time to cap peak VRAM
                for h_idx in range(heads):
                    q_h = q[:, h_idx:h_idx+1, :, :]
                    k_h = k[:, h_idx:h_idx+1, :, :]
                    logits = torch.matmul(
                        q_h.float(), k_h.float().transpose(-2, -1)
                    ) * scale

                    if mask is not None:
                        m = mask
                        if m.ndim == 2:
                            m = m.unsqueeze(0).unsqueeze(0)
                        elif m.ndim == 3:
                            m = m.unsqueeze(1)
                        logits = logits + m

                    aw_h = torch.softmax(logits, dim=-1)  # (B, 1, sq, sk)
                    aw_h = aw_h.mean(dim=0).squeeze(0)    # (sq, sk)

                    attn_received[h_idx] = aw_h.mean(dim=0)  # mean over queries
                    attn_given[h_idx] = aw_h.mean(dim=1)     # mean over keys
                    sq_accum[h_idx] = (aw_h ** 2).mean()

                    del logits, aw_h, q_h, k_h

                head_norms = (sq_accum * seq_len * seq_len).sqrt()

                layer_idx = capture._call_count
                capture.layer_stats[layer_idx] = {
                    "attn_received": attn_received.cpu(),
                    "attn_given": attn_given.cpu(),
                    "head_norms": head_norms.cpu(),
                    "seq_len": seq_len,
                    "n_heads": heads,
                }
                capture._call_count += 1

            # Compute output via the efficient SDPA path (no manual QKT)
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
            return out

        attn_mod.sdpa_attention = patched_sdpa_attention
        self._active = True
        print("[AttentionCapture] Installed monkey-patch on sdpa_attention")

    def remove(self) -> None:
        """Restore original sdpa_attention."""
        if self._original_fn is not None:
            import futudiffu.attention as attn_mod
            attn_mod.sdpa_attention = self._original_fn
            # Also update the import in diffusion_model
            import futudiffu.diffusion_model as dm_mod
            dm_mod.sdpa_attention = self._original_fn
            self._original_fn = None
            self._active = False
            print("[AttentionCapture] Removed monkey-patch")

    def reset(self) -> None:
        """Clear captured stats for next forward pass."""
        self._call_count = 0
        self.layer_stats = {}

    def get_stats(self) -> dict[int, dict]:
        """Return captured stats and reset."""
        stats = dict(self.layer_stats)
        self.reset()
        return stats

    def capture_forward(
        self,
        model,
        latent: Tensor,
        timestep: Tensor,
        conditioning: Tensor,
        num_tokens: int,
        rope_cache: dict,
    ) -> dict[int, dict]:
        """Run a forward pass and return per-layer attention stats.

        Convenience method that handles reset + forward + stats extraction
        in one call.

        Args:
            model: NextDiT model (not compiled -- monkey-patch won't work
                through compiled graph).
            latent: (B, C, H, W) noisy latent.
            timestep: (B,) timestep values.
            conditioning: (B, seq, cap_feat_dim) text encoder output.
            num_tokens: Number of text tokens.
            rope_cache: Precomputed RoPE cache.

        Returns:
            Dict mapping layer_index -> {attn_received, attn_given, head_norms,
            seq_len, n_heads}.
        """
        self.reset()
        with torch.inference_mode():
            model(
                latent, timestep, conditioning,
                num_tokens=num_tokens, rope_cache=rope_cache,
            )
        return self.get_stats()

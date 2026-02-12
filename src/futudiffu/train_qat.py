"""QAT (Quantization-Aware Training) script for Z-Image NextDiT LoRA adapters.

Trains LoRA adapters so that FP8/INT8 quantized attention (SageAttention)
produces final latents matching full-precision SDPA output.

Two training modes:
  direct:    Backprop through checkpointed euler steps. MSE(sage_latent, sdpa_latent).
  reinforce: GRPO with sparse step sampling. K rollouts per group, cosine reward.

Usage:
  python train_qat.py --diffusion-model PATH --text-encoder PATH --vae PATH \\
      --mode direct --num-iterations 5 --lr 1e-4
"""

import math
import random
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LASER_SHARK_PROMPT = (
    'ahem.\n*ting ting ting ting ting*\n'
    'the query model for this is a LARGE LANGUAGE MODEL, specifically QWEN-3-4B, '
    'a GENERAL PURPOSE SEMANTIC PARSER which is able to WRITE SENTENCES AT A TIME '
    'when they are participating in dialogue. however, in this situation, they are '
    'being used as a hidden state generator to steer an *image generation model*, '
    'z-image.\n\nqwen-3-4b, draw me an "enormous laser shark for the sega saturn".'
)


@dataclass
class QATConfig:
    """Configuration for QAT training."""

    # Model paths
    diffusion_model_path: str = ""
    text_encoder_path: str = ""
    vae_path: str = ""
    tokenizer_path: str | None = None

    # Training params
    mode: str = "direct"          # "direct" or "reinforce"
    num_iterations: int = 5
    lr: float = 1e-4
    lora_rank: int = 8
    lora_alpha: int = 16
    grad_clip: float = 1.0

    # Euler params
    steps: int = 30
    cfg: float = 4.0
    width: int = 1280
    height: int = 832
    sampling_shift: float = 1.0
    multiplier: float = 1.0

    # Prompt
    prompt: str = LASER_SHARK_PROMPT
    negative_prompt: str = ""

    # REINFORCE params
    group_size: int = 4           # rollouts per group
    sparse_steps: int = 5         # steps to sample per rollout
    s_churn: float = 0.0          # stochastic churn for diversity

    # Sage attention config
    sage_smooth_k: bool = True
    sage_qk_quant: str = "fp8"
    sage_pv_quant: str = "bf16"

    # Device/dtype
    device: str = "cuda"
    dtype: str = "bfloat16"
    fp8_diffusion: bool = True
    fp8_block_size: int = 128

    # Output
    save_path: str = "lora_qat.safetensors"


def _get_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


# ---------------------------------------------------------------------------
# LoRA injection (self-contained, no external lora.py dependency)
#
# Injects low-rank adapters on all nn.Linear layers in the diffusion model's
# main transformer blocks (layers.*). Freezes all base weights.
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Low-rank adapter wrapping a frozen linear layer.

    out = base(x) + (x @ A^T @ B^T) * (alpha / rank)

    A is initialized Kaiming uniform, B is initialized to zero, so the
    adapter starts as identity (zero contribution).
    """

    def __init__(self, base: nn.Module, rank: int, alpha: int):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        # Determine dimensions from the base module. Works for both
        # nn.Linear and FP8Linear (which exposes in_features/out_features).
        in_features = base.in_features
        out_features = base.out_features

        # Determine dtype: use base weight dtype if it's a standard float,
        # otherwise default to bfloat16 (e.g. when base weight is FP8).
        base_dtype = torch.bfloat16
        if hasattr(base, 'weight') and base.weight is not None:
            if base.weight.dtype in (torch.float16, torch.bfloat16, torch.float32):
                base_dtype = base.weight.dtype

        self.lora_A = nn.Parameter(
            torch.empty(rank, in_features, dtype=base_dtype, device="cuda")
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank, dtype=base_dtype, device="cuda")
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        if not self.enabled:
            return base_out
        # LoRA path: x @ A^T @ B^T * scale
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
        return base_out + lora_out


def inject_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: int = 16,
    target_prefix: str = "layers.",
) -> dict[str, LoRALinear]:
    """Inject LoRA adapters into all Linear/FP8Linear layers under target_prefix.

    Freezes all base model parameters. Returns dict of {name: LoRALinear}.
    """
    from .fp8 import FP8Linear

    # First freeze everything
    for param in model.parameters():
        param.requires_grad_(False)

    lora_modules: dict[str, LoRALinear] = {}

    def _inject(parent: nn.Module, prefix: str):
        for name, child in list(parent.named_children()):
            full_name = f"{prefix}{name}" if prefix else name

            if full_name.startswith(target_prefix) and isinstance(child, (nn.Linear, FP8Linear)):
                lora_layer = LoRALinear(child, rank=rank, alpha=alpha)
                setattr(parent, name, lora_layer)
                lora_modules[full_name] = lora_layer
            else:
                _inject(child, f"{full_name}.")

    _inject(model, "")
    return lora_modules


def get_lora_params(model: nn.Module):
    """Yield all LoRA adapter parameters (A and B matrices)."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            yield module.lora_A
            yield module.lora_B


def set_lora_enabled(model: nn.Module, enabled: bool):
    """Enable or disable all LoRA adapters in the model."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.enabled = enabled


# ---------------------------------------------------------------------------
# Model loading (adapted from generate.py)
# ---------------------------------------------------------------------------

def load_models(config: QATConfig):
    """Load text encoder (encode prompt, then free) and diffusion model.

    Returns:
        (diff_model, positive_cond, negative_cond)
    """
    device = torch.device(config.device)
    dtype = _get_dtype(config.dtype)

    # --- Text encoder ---
    print("[load] Loading text encoder...")
    from .text_encoder import create_tokenizer, encode_prompt, load_text_encoder

    tokenizer = create_tokenizer(config.tokenizer_path)
    te_model = load_text_encoder(config.text_encoder_path, device=device, dtype=dtype)

    print("[load] Encoding positive prompt...")
    positive_cond = encode_prompt(te_model, tokenizer, config.prompt, device=device)
    print(f"  Positive: {positive_cond.shape}")

    print("[load] Encoding negative prompt...")
    negative_cond = encode_prompt(te_model, tokenizer, config.negative_prompt, device=device)
    print(f"  Negative: {negative_cond.shape}")

    # Free text encoder
    del te_model, tokenizer
    torch.cuda.empty_cache()

    # --- Diffusion model ---
    print("[load] Loading diffusion model...")

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

        diff_sd = load_file(config.diffusion_model_path, device=str(device))
        remapped = _strip_diffusion_prefix(diff_sd)
        del diff_sd

        n_layers = _detect_n_layers(remapped.keys())
        cap_feat_dim = _detect_cap_feat_dim(remapped)
        qk_norm = _detect_qk_norm(remapped.keys())
        diff_model = create_diffusion_model(
            dtype=dtype, n_layers=n_layers,
            cap_feat_dim=cap_feat_dim, qk_norm=qk_norm,
        )

        replace_linear_with_fp8(
            diff_model, remapped,
            block_size=config.fp8_block_size, output_dtype=dtype,
        )

        # Load remaining non-FP8 parameters
        remaining = {}
        for k, v in remapped.items():
            if k.endswith(".weight_scale") or k.endswith(".comfy_quant"):
                continue
            remaining[k] = v
        diff_model.load_state_dict(remaining, strict=False, assign=True)
        del remapped, remaining

        diff_model = diff_model.to(device)
    else:
        from .diffusion_model import load_diffusion_model
        diff_model = load_diffusion_model(
            config.diffusion_model_path, device=device, dtype=dtype,
        )

    # Do NOT torch.compile: breaks LoRA + gradient checkpointing
    diff_model.eval()

    return diff_model, positive_cond, negative_cond


# ---------------------------------------------------------------------------
# CFG model_fn builder
# ---------------------------------------------------------------------------

def build_model_fn(
    diff_model: nn.Module,
    positive_cond: torch.Tensor,
    negative_cond: torch.Tensor,
    rope_cache: dict,
    num_tokens: int,
    cfg: float,
    multiplier: float,
):
    """Build a batched CFG model_fn for the euler loop.

    Returns model_fn(x, sigma) -> denoised.
    """
    from .sampling import const_calculate_denoised

    if cfg != 1.0:
        cond_batch = torch.cat([positive_cond, negative_cond], dim=0)  # (2, seq, dim)

    def model_fn(x_in: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        timestep = sigma * multiplier

        if cfg == 1.0:
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
        return denoised_uncond + (denoised_cond - denoised_uncond) * cfg

    return model_fn


# ---------------------------------------------------------------------------
# Attention backend switching
# ---------------------------------------------------------------------------

def switch_to_sdpa():
    """Switch attention to PyTorch SDPA and reset dynamo."""
    from .attention import set_attention_backend
    set_attention_backend("sdpa")
    torch._dynamo.reset()


def switch_to_sage(config: QATConfig):
    """Switch attention to SageAttention and reset dynamo."""
    from .attention import set_attention_backend
    from .sage_attention import configure_sage
    set_attention_backend("sage")
    configure_sage(
        smooth_k=config.sage_smooth_k,
        qk_quant=config.sage_qk_quant,
        pv_quant=config.sage_pv_quant,
    )
    torch._dynamo.reset()


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def log_diagnostics(
    iteration: int,
    loss_val: float,
    lora_params_list: list[torch.Tensor],
    t_start: float,
):
    """Print per-iteration training diagnostics."""
    grad_norms = []
    weight_vals = []
    has_nan = False

    for p in lora_params_list:
        if p.grad is not None:
            gn = p.grad.norm().item()
            grad_norms.append(gn)
            if math.isnan(gn):
                has_nan = True
        wn = p.data.abs()
        weight_vals.append((wn.min().item(), wn.max().item()))
        if torch.isnan(p.data).any():
            has_nan = True

    avg_grad = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
    max_grad = max(grad_norms) if grad_norms else 0.0
    w_min = min(v[0] for v in weight_vals) if weight_vals else 0.0
    w_max = max(v[1] for v in weight_vals) if weight_vals else 0.0
    elapsed = time.time() - t_start

    nan_str = " *** NaN DETECTED ***" if has_nan else ""
    if math.isnan(loss_val):
        nan_str = " *** NaN LOSS ***"

    print(
        f"  iter {iteration:3d} | loss={loss_val:.6f} | "
        f"grad_avg={avg_grad:.4e} grad_max={max_grad:.4e} | "
        f"w_range=[{w_min:.4e}, {w_max:.4e}] | "
        f"time={elapsed:.1f}s{nan_str}"
    )
    return not has_nan and not math.isnan(loss_val)


# ---------------------------------------------------------------------------
# Save LoRA weights
# ---------------------------------------------------------------------------

def save_lora(model: nn.Module, path: str):
    """Save LoRA adapter weights to safetensors."""
    from safetensors.torch import save_file

    state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state_dict[f"{name}.lora_A"] = module.lora_A.data.clone().cpu()
            state_dict[f"{name}.lora_B"] = module.lora_B.data.clone().cpu()

    if not state_dict:
        print("[save] No LoRA weights found, nothing to save.")
        return

    save_file(state_dict, path)
    total_bytes = sum(v.numel() * v.element_size() for v in state_dict.values())
    print(f"[save] Saved {len(state_dict)} tensors ({total_bytes / 1024:.1f} KB) to {path}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_qat(config: QATConfig) -> int:
    """Run QAT training. Returns 0 on success (no NaN), 1 on failure."""

    print("=" * 70)
    print("futudiffu QAT Training")
    print("=" * 70)
    print(f"  Mode: {config.mode}")
    print(f"  Iterations: {config.num_iterations}")
    print(f"  LR: {config.lr}")
    print(f"  LoRA rank={config.lora_rank}, alpha={config.lora_alpha}")
    print(f"  Steps: {config.steps}, CFG: {config.cfg}")
    print(f"  Sage: qk={config.sage_qk_quant}, pv={config.sage_pv_quant}, smooth_k={config.sage_smooth_k}")
    print(f"  Resolution: {config.width}x{config.height}")
    print(f"  FP8 diffusion: {config.fp8_diffusion}")
    print(f"  Device: {config.device}, dtype: {config.dtype}")
    if config.mode == "reinforce":
        print(f"  Group size: {config.group_size}, sparse steps: {config.sparse_steps}")
        print(f"  s_churn: {config.s_churn}")
    print()

    t_total = time.time()
    device = torch.device(config.device)
    dtype = _get_dtype(config.dtype)

    # --- Load models ---
    diff_model, positive_cond, negative_cond = load_models(config)

    # --- Inject LoRA ---
    print(f"[train] Injecting LoRA (rank={config.lora_rank}, alpha={config.lora_alpha})...")
    lora_modules = inject_lora(diff_model, rank=config.lora_rank, alpha=config.lora_alpha)
    print(f"  Injected {len(lora_modules)} LoRA adapters")
    lora_params_list = list(get_lora_params(diff_model))
    n_trainable = sum(p.numel() for p in lora_params_list)
    print(f"  Trainable parameters: {n_trainable:,} ({n_trainable * 2 / 1024**2:.1f} MB at bf16)")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(lora_params_list, lr=config.lr, betas=(0.9, 0.999))

    # --- Prepare conditioning (pad to same length for batched CFG) ---
    pos_len = positive_cond.shape[1]
    neg_len = negative_cond.shape[1]
    max_len = max(pos_len, neg_len)
    if pos_len < max_len:
        positive_cond = F.pad(positive_cond, (0, 0, 0, max_len - pos_len))
    if neg_len < max_len:
        negative_cond = F.pad(negative_cond, (0, 0, 0, max_len - neg_len))
    num_tokens = max_len

    # --- Precompute RoPE cache ---
    latent_h = config.height // 8
    latent_w = config.width // 8
    padded_h = latent_h + ((-latent_h) % diff_model.patch_size)
    padded_w = latent_w + ((-latent_w) % diff_model.patch_size)
    rope_cache = diff_model.prepare_rope_cache(padded_h, padded_w, num_tokens, device)

    # --- Sigma schedule ---
    from .sampling import (
        build_sigmas,
        const_inverse_noise_scaling,
        const_noise_scaling,
        sample_euler,
        sample_euler_train,
        simple_scheduler,
        sparse_step_loss,
    )
    sigma_table = build_sigmas(shift=config.sampling_shift, multiplier=config.multiplier * 1000)
    sigmas = simple_scheduler(sigma_table, config.steps)
    sigmas = sigmas.to(device=device, dtype=dtype)
    print(f"  Sigmas: {sigmas.shape} range [{sigmas[0]:.6f}, {sigmas[-1]:.6f}]")

    # --- Build model_fn ---
    model_fn = build_model_fn(
        diff_model, positive_cond, negative_cond,
        rope_cache, num_tokens, config.cfg, config.multiplier,
    )

    # ===================================================================
    # Training loop
    # ===================================================================
    all_ok = True

    if config.mode == "direct":
        # ---------------------------------------------------------------
        # Mode 1: Direct backprop
        #   For each iteration:
        #   1. Generate reference latent with SDPA (no grad, LoRA off)
        #   2. Generate quantized latent with Sage (with grad, LoRA on)
        #   3. loss = MSE(quantized, reference)
        #   4. Backward through checkpointed euler steps, update LoRA
        # ---------------------------------------------------------------
        print(f"\n[train] Direct mode: {config.num_iterations} iterations, "
              f"{config.steps} euler steps each")

        for iteration in range(config.num_iterations):
            t_start = time.time()
            seed = random.randint(0, 2**63 - 1)

            # Generate noise
            generator = torch.Generator(device=device).manual_seed(seed)
            noise = torch.randn(
                1, 16, latent_h, latent_w,
                dtype=dtype, generator=generator, device=device,
            )
            latent = torch.zeros(1, 16, latent_h, latent_w, device=device, dtype=dtype)
            x_init = const_noise_scaling(sigmas[0], noise, latent)

            # --- Reference pass: SDPA, LoRA disabled, inference mode ---
            set_lora_enabled(diff_model, False)
            switch_to_sdpa()

            with torch.inference_mode():
                x_ref = sample_euler(model_fn, x_init.clone(), sigmas)
                x_ref = const_inverse_noise_scaling(sigmas[-1], x_ref)
            ref_latent = x_ref.detach().clone()
            del x_ref

            # --- Quantized pass: Sage, LoRA enabled, grad mode ---
            set_lora_enabled(diff_model, True)
            switch_to_sage(config)

            x_train = x_init.clone().requires_grad_(True)
            x_out, checkpoints = sample_euler_train(
                model_fn, x_train, sigmas, s_churn=config.s_churn,
            )
            x_out = const_inverse_noise_scaling(sigmas[-1], x_out)

            # --- Loss + backward ---
            loss = F.mse_loss(x_out, ref_latent)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params_list, config.grad_clip)
            optimizer.step()

            # --- Diagnostics ---
            ok = log_diagnostics(iteration, loss.item(), lora_params_list, t_start)
            if not ok:
                all_ok = False

            del x_train, x_out, checkpoints, loss, noise, latent, x_init, ref_latent
            torch.cuda.empty_cache()

    elif config.mode == "reinforce":
        # ---------------------------------------------------------------
        # Mode 2: REINFORCE with sparse step sampling (GRPO)
        #   For each iteration:
        #   1. Generate SDPA reference (LoRA off, inference mode)
        #   2. K rollouts with Sage + LoRA (inference mode, different seeds)
        #   3. Reward = cosine_sim(rollout_latent, reference_latent)
        #   4. Advantage = reward_k - mean(rewards)
        #   5. For rollouts with positive advantage, sample sparse steps
        #   6. sparse_step_loss weighted by advantage -> backward -> step
        # ---------------------------------------------------------------
        K = config.group_size
        print(f"\n[train] REINFORCE mode: {config.num_iterations} iterations, "
              f"K={K} rollouts/group, {config.sparse_steps} sparse steps")

        for iteration in range(config.num_iterations):
            t_start = time.time()
            base_seed = random.randint(0, 2**63 - 1 - K)
            latent = torch.zeros(1, 16, latent_h, latent_w, device=device, dtype=dtype)

            # --- Reference pass ---
            set_lora_enabled(diff_model, False)
            switch_to_sdpa()

            generator = torch.Generator(device=device).manual_seed(base_seed)
            noise_ref = torch.randn(
                1, 16, latent_h, latent_w,
                dtype=dtype, generator=generator, device=device,
            )
            x_ref = const_noise_scaling(sigmas[0], noise_ref, latent)
            with torch.inference_mode():
                x_ref = sample_euler(model_fn, x_ref, sigmas)
                x_ref = const_inverse_noise_scaling(sigmas[-1], x_ref)
            ref_latent = x_ref.detach().clone()
            del x_ref, noise_ref

            # --- K rollouts ---
            set_lora_enabled(diff_model, True)
            switch_to_sage(config)

            rollout_latents = []
            rollout_checkpoints = []

            for k in range(K):
                generator = torch.Generator(device=device).manual_seed(base_seed + k)
                noise_k = torch.randn(
                    1, 16, latent_h, latent_w,
                    dtype=dtype, generator=generator, device=device,
                )
                x_k = const_noise_scaling(sigmas[0], noise_k, latent)
                del noise_k

                with torch.inference_mode():
                    x_out_k, ckpts_k = sample_euler_train(
                        model_fn, x_k, sigmas, s_churn=config.s_churn,
                    )
                    x_out_k = const_inverse_noise_scaling(sigmas[-1], x_out_k)

                rollout_latents.append(x_out_k.detach().clone())
                rollout_checkpoints.append([c.detach().clone() for c in ckpts_k])
                del x_k, x_out_k, ckpts_k

            # --- Compute rewards ---
            ref_flat = ref_latent.flatten().unsqueeze(0)
            rewards = []
            for lat in rollout_latents:
                cos_sim = F.cosine_similarity(
                    lat.flatten().unsqueeze(0), ref_flat, dim=1,
                ).item()
                rewards.append(cos_sim)

            mean_reward = sum(rewards) / len(rewards)
            advantages = [r - mean_reward for r in rewards]
            print(f"  iter {iteration} rewards: "
                  f"{[f'{r:.4f}' for r in rewards]} mean={mean_reward:.4f}")

            # --- Sparse step policy gradient ---
            optimizer.zero_grad()
            total_loss = 0.0
            n_grads = 0

            for k in range(K):
                if advantages[k] <= 0:
                    continue

                step_indices = sorted(random.sample(
                    range(config.steps),
                    min(config.sparse_steps, config.steps),
                ))

                step_loss = sparse_step_loss(
                    rollout_checkpoints[k], model_fn, sigmas, step_indices,
                )
                weighted_loss = step_loss * advantages[k]
                weighted_loss.backward()

                total_loss += weighted_loss.item()
                n_grads += 1
                del step_loss, weighted_loss

            if n_grads > 0:
                # Average gradients across contributing rollouts
                for p in lora_params_list:
                    if p.grad is not None:
                        p.grad.div_(n_grads)
                torch.nn.utils.clip_grad_norm_(lora_params_list, config.grad_clip)
                optimizer.step()

            # --- Diagnostics ---
            loss_val = total_loss / max(n_grads, 1)
            ok = log_diagnostics(iteration, loss_val, lora_params_list, t_start)
            if not ok:
                all_ok = False

            del rollout_latents, rollout_checkpoints, ref_latent
            torch.cuda.empty_cache()

    else:
        raise ValueError(f"Unknown mode: {config.mode!r}")

    # ===================================================================
    # Finish
    # ===================================================================
    elapsed = time.time() - t_total

    # Save LoRA weights
    save_lora(diff_model, config.save_path)

    print(f"\n{'=' * 70}")
    print(f"Training complete in {elapsed:.1f}s")
    print(f"Result: {'PASS (no NaN)' if all_ok else 'FAIL (NaN detected)'}")
    print(f"{'=' * 70}")

    return 0 if all_ok else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="futudiffu QAT: train LoRA adapters for quantized attention fidelity"
    )
    parser.add_argument("--diffusion-model", required=True,
                        help="Path to diffusion model safetensors")
    parser.add_argument("--text-encoder", required=True,
                        help="Path to text encoder safetensors")
    parser.add_argument("--vae", required=True,
                        help="Path to VAE safetensors (unused, for interface compat)")
    parser.add_argument("--tokenizer", default=None,
                        help="Path to tokenizer directory")
    parser.add_argument("--mode", default="direct",
                        choices=["direct", "reinforce"])
    parser.add_argument("--num-iterations", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=832)
    parser.add_argument("--prompt", default=LASER_SHARK_PROMPT)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--group-size", type=int, default=4,
                        help="Rollouts per REINFORCE group")
    parser.add_argument("--sparse-steps", type=int, default=5,
                        help="Steps sampled per REINFORCE rollout")
    parser.add_argument("--s-churn", type=float, default=0.0,
                        help="Stochastic churn for rollout diversity")
    parser.add_argument("--sage-smooth-k",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sage-qk-quant", default="fp8",
                        choices=["fp8", "int8"])
    parser.add_argument("--sage-pv-quant", default="bf16",
                        choices=["bf16", "fp8"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--fp8-diffusion", action="store_true",
                        help="Use FP8 blockwise diffusion model")
    parser.add_argument("--fp8-block-size", type=int, default=128)
    parser.add_argument("--save-path", default="lora_qat.safetensors",
                        help="Output path for LoRA weights")

    args = parser.parse_args()

    config = QATConfig(
        diffusion_model_path=args.diffusion_model,
        text_encoder_path=args.text_encoder,
        vae_path=args.vae,
        tokenizer_path=args.tokenizer,
        mode=args.mode,
        num_iterations=args.num_iterations,
        lr=args.lr,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        grad_clip=args.grad_clip,
        steps=args.steps,
        cfg=args.cfg,
        width=args.width,
        height=args.height,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        group_size=args.group_size,
        sparse_steps=args.sparse_steps,
        s_churn=args.s_churn,
        sage_smooth_k=args.sage_smooth_k,
        sage_qk_quant=args.sage_qk_quant,
        sage_pv_quant=args.sage_pv_quant,
        device=args.device,
        dtype=args.dtype,
        fp8_diffusion=args.fp8_diffusion,
        fp8_block_size=args.fp8_block_size,
        save_path=args.save_path,
    )

    sys.exit(train_qat(config))


if __name__ == "__main__":
    main()

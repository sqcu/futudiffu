"""smoke_test_btrm_policy.py — Real-model BTRM + policy optimization smoke test.

Phase A: Train a BTRM head for 5 steps
  - At each step, probe a different sigma (timestep) from the schedule
  - Positive: SDPA attention hidden states (full precision)
  - Negative: Sage FP8 attention hidden states (quantized)
  - Bradley-Terry loss trains the head to rank SDPA > Sage

Phase B: Optimize LoRA policy for 5 steps using BTRM reward
  - Inject LoRA into all attention+FFN layers (layers.*)
  - At each step, run Sage+LoRA forward, score with frozen BTRM
  - Loss = -BTRM_score (maximize bit_quality reward)
  - Backward flows through sage_attn_op (has autograd backward kernels)
    -> all LoRA params (qkv, out, FFN) get gradients

Logs gradients, weight deltas, and all scalar metrics at every step.
"""

import math
import sys
import time
from collections import OrderedDict

sys.path.insert(0, r"F:\dox\repos\ai\futudiffu\src")

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIFF_MODEL_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\diffusion_models\z_image_fp8_blockwise.safetensors"
TE_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\text_encoders\qwen_3_4b.safetensors"
TOKENIZER_PATH = r"F:\dox\repos\ai\futudiffu\src\futudiffu\tokenizer"

PROMPT = (
    'ahem.\n*ting ting ting ting ting*\n'
    'the query model for this is a LARGE LANGUAGE MODEL, specifically QWEN-3-4B, '
    'a GENERAL PURPOSE SEMANTIC PARSER which is able to WRITE SENTENCES AT A TIME '
    'when they are participating in dialogue. however, in this situation, they are '
    'being used as a hidden state generator to steer an *image generation model*, '
    'z-image.\n\nqwen-3-4b, draw me an "enormous laser shark for the sega saturn".'
)

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
MULTIPLIER = 1.0
SAMPLING_SHIFT = 1.0
STEPS = 30
WIDTH = 1280
HEIGHT = 832
LATENT_H = HEIGHT // 8   # 104
LATENT_W = WIDTH // 8    # 160

# BTRM config
BTRM_LR = 1e-3
BTRM_LOGSQ_WEIGHT = 0.1
BTRM_STEPS = 5

# Policy config
LORA_RANK = 8
LORA_ALPHA = 16
POLICY_LR = 1e-4
POLICY_STEPS = 5
GRAD_CLIP = 1.0


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def log_param_stats(
    label: str,
    params: OrderedDict[str, nn.Parameter],
    prev_snapshot: dict[str, torch.Tensor] | None = None,
) -> None:
    """Print gradient norms and weight deltas for a set of named parameters."""
    for pname, param in params.items():
        grad_norm = param.grad.norm().item() if param.grad is not None else 0.0
        w_norm = param.data.norm().item()
        w_max = param.data.abs().max().item()

        delta = 0.0
        if prev_snapshot is not None and pname in prev_snapshot:
            delta = (param.data - prev_snapshot[pname]).norm().item()

        flags = ""
        if torch.isnan(param.data).any():
            flags = " *** NaN WEIGHT ***"
        elif param.grad is not None and torch.isnan(param.grad).any():
            flags = " *** NaN GRAD ***"
        elif param.grad is not None and param.grad.abs().sum().item() == 0:
            flags = " [zero grad]"

        short_name = pname.rsplit(".", 1)[-1] if len(pname) > 50 else pname
        print(
            f"    {pname:55s} | grad={grad_norm:.3e} "
            f"w_norm={w_norm:.3e} w_max={w_max:.3e} delta={delta:.3e}{flags}"
        )


def snapshot_params(params: OrderedDict[str, nn.Parameter]) -> dict[str, torch.Tensor]:
    return {name: p.data.clone() for name, p in params.items()}


# ---------------------------------------------------------------------------
# Hidden state capture via forward hook
# ---------------------------------------------------------------------------

class HiddenCapture:
    """Hook on the last transformer block to capture hidden states."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.captured: torch.Tensor | None = None
        self._handle = None

    def install(self) -> None:
        if self._handle is not None:
            return
        last_block = self.model.layers[-1]

        def hook(_module, _input, output):
            self.captured = output

        self._handle = last_block.register_forward_hook(hook)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def get(self) -> torch.Tensor:
        """Return captured tensor (still in compute graph) and clear storage."""
        h = self.captured
        self.captured = None
        assert h is not None, "Hook did not fire — check model.layers"
        return h


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models():
    """Load TE (encode prompt then free), diffusion model, and precompute caches.

    Returns (diff_model, pos_cond, sigmas, rope_cache, num_tokens).
    """
    from futudiffu.text_encoder import create_tokenizer, encode_prompt, load_text_encoder
    from futudiffu.diffusion_model import (
        create_diffusion_model,
        _detect_cap_feat_dim,
        _detect_n_layers,
        _detect_qk_norm,
        _strip_diffusion_prefix,
    )
    from futudiffu.fp8 import replace_linear_with_fp8
    from safetensors.torch import load_file
    from futudiffu.sampling import build_sigmas, simple_scheduler

    # --- Text encoder ---
    print("[load] Text encoder (BF16)...")
    tokenizer = create_tokenizer(TOKENIZER_PATH)
    te = load_text_encoder(TE_PATH, device=DEVICE, dtype=DTYPE)
    te = torch.compile(te, mode="default")

    print("[load] Encoding prompt...")
    pos_cond = encode_prompt(te, tokenizer, PROMPT, device=DEVICE)
    neg_cond = encode_prompt(te, tokenizer, "", device=DEVICE)
    print(f"  pos: {pos_cond.shape}, neg: {neg_cond.shape}")

    del te, tokenizer
    torch.cuda.empty_cache()

    # --- Diffusion model (FP8) ---
    print("[load] Diffusion model (FP8 blockwise)...")
    diff_sd = load_file(DIFF_MODEL_PATH, device=str(DEVICE))
    remapped = _strip_diffusion_prefix(diff_sd)
    del diff_sd

    n_layers = _detect_n_layers(remapped.keys())
    cap_feat_dim = _detect_cap_feat_dim(remapped)
    qk_norm = _detect_qk_norm(remapped.keys())

    model = create_diffusion_model(
        dtype=DTYPE, n_layers=n_layers,
        cap_feat_dim=cap_feat_dim, qk_norm=qk_norm,
    )
    replace_linear_with_fp8(model, remapped, block_size=128, output_dtype=DTYPE)

    remaining = {
        k: v for k, v in remapped.items()
        if not k.endswith(".weight_scale") and not k.endswith(".comfy_quant")
    }
    model.load_state_dict(remaining, strict=False, assign=True)
    del remapped, remaining

    model = model.to(DEVICE)
    model.eval()

    # --- Sigmas ---
    sigma_table = build_sigmas(shift=SAMPLING_SHIFT, multiplier=MULTIPLIER * 1000)
    sigmas = simple_scheduler(sigma_table, STEPS)
    sigmas = sigmas.to(device=DEVICE, dtype=DTYPE)

    # --- Pad conditioning to same length ---
    max_len = max(pos_cond.shape[1], neg_cond.shape[1])
    if pos_cond.shape[1] < max_len:
        pos_cond = F.pad(pos_cond, (0, 0, 0, max_len - pos_cond.shape[1]))
    if neg_cond.shape[1] < max_len:
        neg_cond = F.pad(neg_cond, (0, 0, 0, max_len - neg_cond.shape[1]))
    num_tokens = max_len

    # --- RoPE cache (clone to escape any inference_mode context) ---
    padded_h = LATENT_H + ((-LATENT_H) % model.patch_size)
    padded_w = LATENT_W + ((-LATENT_W) % model.patch_size)
    rope_cache = model.prepare_rope_cache(padded_h, padded_w, num_tokens, DEVICE)
    rope_cache = {k: v.clone() if isinstance(v, torch.Tensor) else v
                  for k, v in rope_cache.items()}

    # Clone conditioning to escape inference_mode tensors from encode_prompt.
    # Policy optimization needs these in the autograd graph.
    pos_cond = pos_cond.clone()
    neg_cond = neg_cond.clone()

    # We only use positive conditioning for BTRM/policy (no CFG needed)
    # CFG is a sampling-time concern; BTRM evaluates single-forward quality.
    print(f"[load] Complete: dim={model.dim}, layers={n_layers}, "
          f"cap_feat_dim={cap_feat_dim}, qk_norm={qk_norm}")
    print(f"  Sigmas: {sigmas.shape} range [{sigmas[0]:.4f}, {sigmas[-1]:.4f}]")
    print(f"  Conditioning: {pos_cond.shape}")

    mem = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  Peak GPU memory after load: {mem:.1f} GB")

    return model, pos_cond, sigmas, rope_cache, num_tokens


# ---------------------------------------------------------------------------
# Phase A: BTRM Head Training
# ---------------------------------------------------------------------------

def train_btrm(model, pos_cond, sigmas, rope_cache, num_tokens):
    """Train BTRM head for BTRM_STEPS steps. Returns trained head + log."""
    from futudiffu.attention import set_attention_backend
    from futudiffu.sage_attention import configure_sage
    from futudiffu.btrm import BTRMHead, bradley_terry_loss, logsquare_regularizer

    log_section("Phase A: BTRM Head Training (5 steps)")

    # Create BTRM head
    btrm = BTRMHead(
        hidden_dim=model.dim,
        head_names=("bit_quality", "step_quality"),
        logit_cap=10.0,
    ).to(device=DEVICE, dtype=DTYPE)

    optimizer = torch.optim.AdamW(btrm.parameters(), lr=BTRM_LR)
    capture = HiddenCapture(model)
    capture.install()

    # Named params for logging
    btrm_params = OrderedDict(
        (name, p) for name, p in btrm.named_parameters() if p.requires_grad
    )
    n_params = sum(p.numel() for p in btrm_params.values())
    print(f"BTRM head: {n_params:,} trainable parameters")
    for name, p in btrm_params.items():
        print(f"  {name}: {list(p.shape)}")

    # Pick 5 sigma values spread across the schedule (early to late denoising)
    n_sigmas = len(sigmas) - 1  # exclude terminal 0.0
    sigma_indices = [int(i * (n_sigmas - 1) / (BTRM_STEPS - 1)) for i in range(BTRM_STEPS)]
    print(f"\nProbe sigmas: {[(i, sigmas[i].item()) for i in sigma_indices]}")

    # Configure Sage (FP8 QK + BF16 PV with K-smoothing)
    configure_sage(smooth_k=True, qk_quant="fp8", pv_quant="bf16")

    log = []

    for step in range(BTRM_STEPS):
        t0 = time.time()
        prev = snapshot_params(btrm_params)

        sigma_idx = sigma_indices[step]
        sigma = sigmas[sigma_idx]
        timestep = (sigma * MULTIPLIER).unsqueeze(0)

        # Random noise -> noisy latent (clean_latent=0, so x_t = sigma * noise)
        noise = torch.randn(1, 16, LATENT_H, LATENT_W, dtype=DTYPE, device=DEVICE)
        x_t = sigma * noise

        # --- Positive sample: SDPA (full precision attention) ---
        set_attention_backend("sdpa")
        with torch.no_grad():
            model(x_t, timestep, pos_cond,
                  num_tokens=num_tokens, rope_cache=rope_cache)
        hidden_pos = capture.get().detach()

        # --- Negative sample: Sage FP8 (quantized attention) ---
        set_attention_backend("sage")
        with torch.no_grad():
            model(x_t, timestep, pos_cond,
                  num_tokens=num_tokens, rope_cache=rope_cache)
        hidden_neg = capture.get().detach()

        # --- Hidden state divergence (diagnostic) ---
        with torch.no_grad():
            h_cos = F.cosine_similarity(
                hidden_pos.flatten().unsqueeze(0),
                hidden_neg.flatten().unsqueeze(0),
            ).item()

        # --- Score + loss ---
        pos_scores = btrm(hidden_pos)  # (1, 2)
        neg_scores = btrm(hidden_neg)  # (1, 2)

        bt_loss = bradley_terry_loss(pos_scores[:, 0], neg_scores[:, 0])
        logsq = logsquare_regularizer(pos_scores[:, 0])
        loss = bt_loss + BTRM_LOGSQ_WEIGHT * logsq

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        dt = time.time() - t0
        margin = (pos_scores[:, 0] - neg_scores[:, 0]).item()

        entry = dict(
            step=step,
            sigma=sigma.item(),
            loss=loss.item(),
            bt_loss=bt_loss.item(),
            logsq=logsq.item(),
            pos_score=pos_scores.detach().cpu().tolist()[0],
            neg_score=neg_scores.detach().cpu().tolist()[0],
            margin=margin,
            hidden_cos=h_cos,
            time=dt,
        )
        log.append(entry)

        is_nan = math.isnan(loss.item())
        flag = " *** NaN LOSS ***" if is_nan else ""
        print(
            f"  step {step} | sigma={sigma.item():.4f} | "
            f"loss={loss.item():.4f} (bt={bt_loss.item():.4f} logsq={logsq.item():.4f}) | "
            f"margin={margin:+.4f} | hidden_cos={h_cos:.6f} | {dt:.1f}s{flag}"
        )
        print(f"    pos_scores={entry['pos_score']}  neg_scores={entry['neg_score']}")
        log_param_stats("BTRM", btrm_params, prev)

        del noise, x_t, hidden_pos, hidden_neg, pos_scores, neg_scores
        torch.cuda.empty_cache()

    capture.remove()

    print(f"\nBTRM training summary:")
    print(f"  Margin trajectory: {[f'{e['margin']:+.4f}' for e in log]}")
    print(f"  Positive = SDPA preferred when margin > 0")

    return btrm, log


# ---------------------------------------------------------------------------
# Gradient-checkpointed model forward
# ---------------------------------------------------------------------------

def forward_checkpointed(model, x, timesteps, context, num_tokens, rope_cache):
    """Model forward with per-block gradient checkpointing on the 30 main layers.

    Memory: ~1GB (30 layer-boundary tensors) + ~300MB peak (one layer recompute)
    vs ~12GB without checkpointing. Fits comfortably on a 24GB card.

    Embedding + refiners run without grad (no LoRA targets there that we care
    about for the smoke test). Main layers are checkpointed. Final layer runs
    normally.

    Returns the last main-layer output (before final_layer) for BTRM scoring,
    plus the full model output for diagnostics.
    """
    from torch.utils.checkpoint import checkpoint as grad_ckpt
    from futudiffu.diffusion_model import pad_to_patch_size, pad_zimage

    # --- Phase 1: Embedding + refiners (no grad, cheap) ---
    with torch.no_grad():
        t = 1 - timesteps
        bs, c, h, w = x.shape
        x_padded = pad_to_patch_size(x, (model.patch_size, model.patch_size))

        t_emb = model.t_embedder(t * model.time_scale, dtype=x.dtype)
        adaln_input = t_emb

        bsz = x_padded.shape[0]
        pH = pW = model.patch_size

        cap_feats_embedded = model.cap_embedder(context)
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

        img_len = x_patches.shape[1]

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

        l_effective_cap_len = [embed.shape[1] - img_len] * bsz
        img_sizes = [(H, W)] * bsz

    # --- Phase 2: Detach and start autograd graph for checkpointed layers ---
    # Everything before this point is constant w.r.t. LoRA (no grad needed).
    embed = embed.detach().clone().requires_grad_(True)
    adaln_input = adaln_input.detach().clone()
    freqs_cis = freqs_cis.detach().clone()

    # --- Phase 3: 30 main layers with per-block gradient checkpointing ---
    # Each layer's activations are discarded and recomputed during backward.
    # Memory: ~33MB per boundary tensor (30 stored) = ~1GB total.
    for layer in model.layers:
        embed = grad_ckpt(
            layer, embed, None, freqs_cis, adaln_input,
            use_reentrant=False,
        )

    # `embed` is now the output of the last transformer block -- this is what
    # the BTRM scores. It has gradient connections to all main-layer LoRA params.
    last_hidden = embed

    # --- Phase 4: Final layer (small, no checkpointing needed) ---
    img = model.final_layer(embed, adaln_input)
    img = model.unpatchify(img, img_sizes, l_effective_cap_len, return_tensor=True)[:, :, :h, :w]

    return -img, last_hidden


# ---------------------------------------------------------------------------
# Phase B: Policy Optimization
# ---------------------------------------------------------------------------

def train_policy(model, btrm, pos_cond, sigmas, rope_cache, num_tokens):
    """Train LoRA adapters for 5 steps using frozen BTRM as reward signal.

    Uses per-block gradient checkpointing to fit the 30-layer backward pass
    in ~8-9GB (vs ~18GB without checkpointing). LoRA params in the 30 main
    layers get gradients; refiner LoRA params get zero grad (run in no_grad
    embedding phase).
    """
    from futudiffu.lora import (
        LoRALinear, inject_lora, get_lora_params,
    )
    from futudiffu.attention import set_attention_backend
    from futudiffu.sage_attention import configure_sage

    log_section("Phase B: LoRA Policy Optimization (5 steps, BTRM reward)")
    print("  Using per-block gradient checkpointing (30 layers)")
    print("  Memory: ~1GB boundary tensors + ~300MB peak recompute")

    # Freeze BTRM head (acts as fixed differentiable reward function)
    btrm.eval()
    for p in btrm.parameters():
        p.requires_grad_(False)

    # Inject LoRA
    print("\n[policy] Injecting LoRA adapters...")
    injected = inject_lora(model, rank=LORA_RANK, alpha=LORA_ALPHA)
    lora_params = list(get_lora_params(model))
    n_trainable = sum(p.numel() for p in lora_params)
    print(f"  {len(injected)} adapters, {n_trainable:,} trainable params "
          f"({n_trainable * 2 / 1024**2:.1f} MB bf16)")

    # Count main-layer vs refiner LoRA
    n_main = sum(1 for name in injected if name.startswith("layers."))
    n_refiner = len(injected) - n_main
    print(f"  Main layers: {n_main} adapters (get gradients)")
    print(f"  Refiners: {n_refiner} adapters (no grad, run in embedding phase)")

    # Pick a sample of LoRA modules for detailed logging
    sample_params = OrderedDict()
    for name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        if any(pat in name for pat in [
            "layers.0.attention.qkv", "layers.0.attention.out",
            "layers.0.feed_forward.w1",
            "layers.29.attention.qkv", "layers.29.attention.out",
            "layers.29.feed_forward.w1",
        ]):
            sample_params[f"{name}.A"] = module.lora_A
            sample_params[f"{name}.B"] = module.lora_B

    print(f"\n  Detailed logging for {len(sample_params)} sample parameters:")
    for name in sample_params:
        print(f"    {name}")

    optimizer = torch.optim.AdamW(lora_params, lr=POLICY_LR, betas=(0.9, 0.999))

    # Lock in Sage backend (no compile = no CUDA graph contamination)
    set_attention_backend("sage")
    configure_sage(smooth_k=True, qk_quant="fp8", pv_quant="bf16")

    # Sigma schedule for probing
    n_sigmas = len(sigmas) - 1
    sigma_indices = [int(i * (n_sigmas - 1) / (POLICY_STEPS - 1)) for i in range(POLICY_STEPS)]
    print(f"\n  Probe sigmas: {[(i, sigmas[i].item()) for i in sigma_indices]}")

    log = []

    for step in range(POLICY_STEPS):
        t0 = time.time()
        prev = snapshot_params(sample_params)

        sigma_idx = sigma_indices[step]
        sigma = sigmas[sigma_idx]
        timestep = (sigma * MULTIPLIER).unsqueeze(0)

        noise = torch.randn(1, 16, LATENT_H, LATENT_W, dtype=DTYPE, device=DEVICE)
        x_t = sigma * noise

        # Checkpointed forward: embedding+refiners (no grad) → 30 main layers
        # (checkpointed) → final layer. Returns model output + last hidden.
        model_output, last_hidden = forward_checkpointed(
            model, x_t, timestep, pos_cond,
            num_tokens=num_tokens, rope_cache=rope_cache,
        )

        # Score last hidden with frozen BTRM head (still differentiable —
        # gradients flow through BTRM projection → hidden → checkpointed
        # layers → sage_attn_op backward → LoRA params)
        scores = btrm(last_hidden)  # (1, 2)
        reward_bq = scores[:, 0].mean()   # bit_quality head
        reward_sq = scores[:, 1].mean()   # step_quality head

        # Maximize bit_quality reward
        loss = -reward_bq

        optimizer.zero_grad()
        loss.backward()

        mem = torch.cuda.max_memory_allocated() / 1024**3
        grad_norm = torch.nn.utils.clip_grad_norm_(lora_params, GRAD_CLIP)
        optimizer.step()

        dt = time.time() - t0

        # Grad statistics
        n_with_grad = sum(
            1 for p in lora_params
            if p.grad is not None and p.grad.abs().sum().item() > 0
        )
        n_zero_grad = sum(
            1 for p in lora_params
            if p.grad is not None and p.grad.abs().sum().item() == 0
        )
        n_total = len(lora_params)

        entry = dict(
            step=step,
            sigma=sigma.item(),
            reward_bq=reward_bq.item(),
            reward_sq=reward_sq.item(),
            loss=loss.item(),
            grad_norm=grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            n_with_grad=n_with_grad,
            n_zero_grad=n_zero_grad,
            n_total=n_total,
            time=dt,
        )
        log.append(entry)

        is_nan = math.isnan(loss.item())
        flag = " *** NaN ***" if is_nan else ""
        print(
            f"  step {step} | sigma={sigma.item():.4f} | "
            f"reward_bq={reward_bq.item():+.4f} reward_sq={reward_sq.item():+.4f} | "
            f"loss={loss.item():.4f} | grad_norm={entry['grad_norm']:.3e} | "
            f"grads: {n_with_grad}/{n_total} nonzero ({n_zero_grad} zero) | "
            f"mem={mem:.1f}GB | {dt:.1f}s{flag}"
        )
        log_param_stats("LoRA", sample_params, prev)

        del noise, x_t, model_output, last_hidden, scores
        torch.cuda.empty_cache()

    print(f"\nPolicy optimization summary:")
    print(f"  Reward trajectory: {[f'{e['reward_bq']:+.4f}' for e in log]}")
    print(f"  Grad norm trajectory: {[f'{e['grad_norm']:.3e}' for e in log]}")

    # Save LoRA state dict
    from futudiffu.lora import lora_state_dict
    sd = lora_state_dict(model)
    save_path = r"F:\dox\repos\ai\futudiffu\smoke_test_lora.safetensors"
    from safetensors.torch import save_file
    save_file({k: v.cpu() for k, v in sd.items()}, save_path)
    total_kb = sum(v.numel() * v.element_size() for v in sd.values()) / 1024
    print(f"\n  Saved {len(sd)} LoRA tensors ({total_kb:.1f} KB) to {save_path}")

    return log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_total = time.time()

    log_section("Smoke Test: 5-step BTRM + 5-step Policy Optimization")
    print(f"BTRM: lr={BTRM_LR}, logsq_weight={BTRM_LOGSQ_WEIGHT}")
    print(f"Policy: lr={POLICY_LR}, lora_rank={LORA_RANK}, lora_alpha={LORA_ALPHA}")
    print(f"Sage: FP8 QK + BF16 PV, smooth_k=True")
    print(f"Resolution: {WIDTH}x{HEIGHT}, steps={STEPS}, dtype=bf16")

    model, pos_cond, sigmas, rope_cache, num_tokens = load_models()

    btrm, btrm_log = train_btrm(model, pos_cond, sigmas, rope_cache, num_tokens)

    policy_log = train_policy(model, btrm, pos_cond, sigmas, rope_cache, num_tokens)

    # --- Final summary ---
    log_section("Final Summary")

    print("BTRM Training (5 steps):")
    print(f"  {'step':>4s}  {'sigma':>6s}  {'loss':>8s}  {'bt_loss':>8s}  "
          f"{'margin':>8s}  {'hidden_cos':>10s}")
    for e in btrm_log:
        print(f"  {e['step']:4d}  {e['sigma']:6.4f}  {e['loss']:8.4f}  "
              f"{e['bt_loss']:8.4f}  {e['margin']:+8.4f}  {e['hidden_cos']:10.6f}")

    print(f"\nPolicy Optimization (5 steps):")
    print(f"  {'step':>4s}  {'sigma':>6s}  {'reward_bq':>10s}  {'reward_sq':>10s}  "
          f"{'grad_norm':>10s}  {'grads':>12s}")
    for e in policy_log:
        print(f"  {e['step']:4d}  {e['sigma']:6.4f}  {e['reward_bq']:+10.4f}  "
              f"{e['reward_sq']:+10.4f}  {e['grad_norm']:10.3e}  "
              f"{e['n_with_grad']}/{e['n_total']}")

    # NaN check
    all_ok = True
    for e in btrm_log:
        if math.isnan(e["loss"]):
            all_ok = False
    for e in policy_log:
        if math.isnan(e["loss"]):
            all_ok = False

    elapsed = time.time() - t_total
    mem = torch.cuda.max_memory_allocated() / 1024**3

    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Peak GPU memory: {mem:.1f} GB")
    print(f"Result: {'PASS (no NaN)' if all_ok else 'FAIL (NaN detected)'}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

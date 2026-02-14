"""smoke_test_btrm_policy.py -- Real-model BTRM + policy optimization smoke test.

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

Uses canonical training_utils for HiddenCapture, forward_checkpointed,
and logging helpers.
"""

import math
import sys
import time
from collections import OrderedDict

sys.path.insert(0, r"F:\dox\repos\ai\futudiffu\src")

import torch
import torch.nn as nn
import torch.nn.functional as F

from futudiffu.training_utils import (
    HiddenCapture,
    forward_checkpointed,
    log_param_stats,
    log_section,
    snapshot_params,
)

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
# Model loading
# ---------------------------------------------------------------------------

def load_models():
    """Load TE (encode prompt then free), diffusion model, and precompute caches."""
    from futudiffu.text_encoder import create_tokenizer, encode_prompt, load_text_encoder
    from futudiffu.diffusion_model import (
        create_diffusion_model, _detect_cap_feat_dim, _detect_n_layers,
        _detect_qk_norm, _strip_diffusion_prefix,
    )
    from futudiffu.fp8 import replace_linear_with_fp8
    from safetensors.torch import load_file
    from futudiffu.training_utils import prepare_conditioning, prepare_latent_state

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

    pos_cond = pos_cond.clone()
    neg_cond = neg_cond.clone()
    pos_cond, neg_cond, num_tokens = prepare_conditioning(pos_cond, neg_cond)

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
    remaining = {k: v for k, v in remapped.items()
                 if not k.endswith(".weight_scale") and not k.endswith(".comfy_quant")}
    model.load_state_dict(remaining, strict=False, assign=True)
    del remapped, remaining
    model = model.to(DEVICE)
    model.eval()

    rope_cache, sigmas, _, _ = prepare_latent_state(
        model, WIDTH, HEIGHT, num_tokens, DEVICE, DTYPE,
        sampling_shift=SAMPLING_SHIFT, multiplier=MULTIPLIER, steps=STEPS,
    )

    print(f"[load] Complete: dim={model.dim}, layers={n_layers}")
    print(f"  Sigmas: {sigmas.shape} range [{sigmas[0]:.4f}, {sigmas[-1]:.4f}]")
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

    btrm = BTRMHead(
        hidden_dim=model.dim,
        head_names=("bit_quality", "step_quality"),
        logit_cap=10.0,
    ).to(device=DEVICE, dtype=DTYPE)

    optimizer = torch.optim.AdamW(btrm.parameters(), lr=BTRM_LR)
    capture = HiddenCapture(model)
    capture.install()

    btrm_params = OrderedDict(
        (name, p) for name, p in btrm.named_parameters() if p.requires_grad
    )
    n_params = sum(p.numel() for p in btrm_params.values())
    print(f"BTRM head: {n_params:,} trainable parameters")

    n_sigmas = len(sigmas) - 1
    sigma_indices = [int(i * (n_sigmas - 1) / (BTRM_STEPS - 1)) for i in range(BTRM_STEPS)]
    print(f"\nProbe sigmas: {[(i, sigmas[i].item()) for i in sigma_indices]}")

    configure_sage(smooth_k=True, qk_quant="fp8", pv_quant="bf16")

    log = []
    for step in range(BTRM_STEPS):
        t0 = time.time()
        prev = snapshot_params(btrm_params)

        sigma_idx = sigma_indices[step]
        sigma = sigmas[sigma_idx]
        timestep = (sigma * MULTIPLIER).unsqueeze(0)

        noise = torch.randn(1, 16, LATENT_H, LATENT_W, dtype=DTYPE, device=DEVICE)
        x_t = sigma * noise

        set_attention_backend("sdpa")
        with torch.no_grad():
            model(x_t, timestep, pos_cond,
                  num_tokens=num_tokens, rope_cache=rope_cache)
        hidden_pos = capture.get().detach()

        set_attention_backend("sage")
        with torch.no_grad():
            model(x_t, timestep, pos_cond,
                  num_tokens=num_tokens, rope_cache=rope_cache)
        hidden_neg = capture.get().detach()

        with torch.no_grad():
            h_cos = F.cosine_similarity(
                hidden_pos.flatten().unsqueeze(0),
                hidden_neg.flatten().unsqueeze(0),
            ).item()

        pos_scores = btrm(hidden_pos)
        neg_scores = btrm(hidden_neg)
        bt_loss = bradley_terry_loss(pos_scores[:, 0], neg_scores[:, 0])
        logsq = logsquare_regularizer(pos_scores[:, 0])
        loss = bt_loss + BTRM_LOGSQ_WEIGHT * logsq

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        dt = time.time() - t0
        margin = (pos_scores[:, 0] - neg_scores[:, 0]).item()

        entry = dict(
            step=step, sigma=sigma.item(), loss=loss.item(),
            bt_loss=bt_loss.item(), logsq=logsq.item(),
            pos_score=pos_scores.detach().cpu().tolist()[0],
            neg_score=neg_scores.detach().cpu().tolist()[0],
            margin=margin, hidden_cos=h_cos, time=dt,
        )
        log.append(entry)

        flag = " *** NaN LOSS ***" if math.isnan(loss.item()) else ""
        print(
            f"  step {step} | sigma={sigma.item():.4f} | "
            f"loss={loss.item():.4f} (bt={bt_loss.item():.4f} logsq={logsq.item():.4f}) | "
            f"margin={margin:+.4f} | hidden_cos={h_cos:.6f} | {dt:.1f}s{flag}"
        )
        log_param_stats("BTRM", btrm_params, prev)

        del noise, x_t, hidden_pos, hidden_neg, pos_scores, neg_scores
        torch.cuda.empty_cache()

    capture.remove()
    print(f"\nBTRM training summary:")
    print(f"  Margin trajectory: {[f'{e['margin']:+.4f}' for e in log]}")
    return btrm, log


# ---------------------------------------------------------------------------
# Phase B: Policy Optimization
# ---------------------------------------------------------------------------

def train_policy(model, btrm, pos_cond, sigmas, rope_cache, num_tokens):
    """Train LoRA adapters for 5 steps using frozen BTRM as reward signal."""
    from futudiffu.lora import LoRALinear, inject_lora, get_lora_params, lora_state_dict
    from futudiffu.attention import set_attention_backend
    from futudiffu.sage_attention import configure_sage

    log_section("Phase B: LoRA Policy Optimization (5 steps, BTRM reward)")

    btrm.eval()
    for p in btrm.parameters():
        p.requires_grad_(False)

    print("\n[policy] Injecting LoRA adapters...")
    injected = inject_lora(model, rank=LORA_RANK, alpha=LORA_ALPHA)
    lora_params = list(get_lora_params(model))
    n_trainable = sum(p.numel() for p in lora_params)
    print(f"  {len(injected)} adapters, {n_trainable:,} trainable params")

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
            for aname, adapter in module.adapters.items():
                sample_params[f"{name}.{aname}.A"] = adapter.lora_A
                sample_params[f"{name}.{aname}.B"] = adapter.lora_B

    optimizer = torch.optim.AdamW(lora_params, lr=POLICY_LR, betas=(0.9, 0.999))

    set_attention_backend("sage")
    configure_sage(smooth_k=True, qk_quant="fp8", pv_quant="bf16")

    n_sigmas = len(sigmas) - 1
    sigma_indices = [int(i * (n_sigmas - 1) / (POLICY_STEPS - 1)) for i in range(POLICY_STEPS)]

    log = []
    for step in range(POLICY_STEPS):
        t0 = time.time()
        prev = snapshot_params(sample_params)

        sigma_idx = sigma_indices[step]
        sigma = sigmas[sigma_idx]
        timestep = (sigma * MULTIPLIER).unsqueeze(0)

        noise = torch.randn(1, 16, LATENT_H, LATENT_W, dtype=DTYPE, device=DEVICE)
        x_t = sigma * noise

        model_output, last_hidden = forward_checkpointed(
            model, x_t, timestep, pos_cond,
            num_tokens=num_tokens, rope_cache=rope_cache,
        )

        scores = btrm(last_hidden)
        reward_bq = scores[:, 0].mean()
        reward_sq = scores[:, 1].mean()
        loss = -reward_bq

        optimizer.zero_grad()
        loss.backward()
        mem = torch.cuda.max_memory_allocated() / 1024**3
        grad_norm = torch.nn.utils.clip_grad_norm_(lora_params, GRAD_CLIP)
        optimizer.step()

        dt = time.time() - t0
        n_with_grad = sum(1 for p in lora_params
                          if p.grad is not None and p.grad.abs().sum().item() > 0)

        entry = dict(
            step=step, sigma=sigma.item(),
            reward_bq=reward_bq.item(), reward_sq=reward_sq.item(),
            loss=loss.item(),
            grad_norm=grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            n_with_grad=n_with_grad, n_total=len(lora_params), time=dt,
        )
        log.append(entry)

        print(
            f"  step {step} | sigma={sigma.item():.4f} | "
            f"reward_bq={reward_bq.item():+.4f} reward_sq={reward_sq.item():+.4f} | "
            f"loss={loss.item():.4f} | grad_norm={entry['grad_norm']:.3e} | "
            f"grads: {n_with_grad}/{len(lora_params)} | mem={mem:.1f}GB | {dt:.1f}s"
        )
        log_param_stats("LoRA", sample_params, prev)

        del noise, x_t, model_output, last_hidden, scores
        torch.cuda.empty_cache()

    print(f"\nPolicy optimization summary:")
    print(f"  Reward trajectory: {[f'{e['reward_bq']:+.4f}' for e in log]}")

    sd = lora_state_dict(model)
    save_path = r"F:\dox\repos\ai\futudiffu\smoke_test_lora.safetensors"
    from safetensors.torch import save_file
    save_file({k: v.cpu() for k, v in sd.items()}, save_path)
    print(f"\n  Saved {len(sd)} LoRA tensors to {save_path}")

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

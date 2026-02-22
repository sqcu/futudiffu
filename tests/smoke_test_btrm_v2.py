"""smoke_test_btrm_v2.py -- BTRM + policy with proper paired training data.

Phase 1: Generate paired data (5 seeds x {SDPA, Sage} x 30-step euler)
  - SDPA renders = positive (full-precision attention)
  - Sage renders = negative (FP8 quantized attention)
  - Saves step latents at 7 evenly-spaced checkpoints + final

Phase 2: Train R_theta = frozen backbone + LoRA("rtheta", layers 28-29) + ScoreUnembedder
  - LoRA adapters give the reward model nonlinear capacity to amplify
    quality-discriminative features beyond what the frozen backbone provides
  - Gradient checkpointing on all 30 layers (modern ML standard)

Phase 3: Policy optimization with stacked adapters
  - Stack "ptheta" adapter on ALL layers (horizontal, same LoRALinear)
  - "rtheta" adapter: scale=0, frozen (contributes nothing during policy forward)
  - "ptheta" adapter: scale=1, trainable (active during policy forward)
  - BTRM head reads hidden states from policy-adapted model
  - Gradient checkpointing on all layers always

Uses canonical training_utils for forward_checkpointed,
prepare_conditioning, prepare_latent_state, and logging helpers.
"""

import math
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, r"F:\dox\repos\ai\futudiffu\src")

import torch
import torch.nn as nn
import torch.nn.functional as F

from futudiffu.training_utils import (
    build_cfg_model_fn,
    forward_checkpointed,
    log_section,
    prepare_conditioning,
    prepare_latent_state,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIFF_MODEL_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\diffusion_models\z_image_fp8_blockwise.safetensors"
TE_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\text_encoders\qwen_3_4b.safetensors"
TOKENIZER_PATH = r"F:\dox\repos\ai\futudiffu\src\futudiffu\tokenizer"
DATA_DIR = r"F:\dox\repos\ai\futudiffu\btrm_data"

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
CFG = 4.0
LATENT_H = HEIGHT // 8   # 104
LATENT_W = WIDTH // 8    # 160

# Data generation
N_SEEDS = 5
BASE_SEED = 42000
CKPT_STEPS = [0, 5, 10, 15, 20, 25, 29]

# BTRM config
BTRM_LR = 1e-3
BTRM_LOGSQ_WEIGHT = 0.1
BTRM_EPOCHS = 3

# Policy config
LORA_RANK = 8
LORA_ALPHA = 16
POLICY_LR = 1e-4
POLICY_STEPS = 5
GRAD_CLIP = 1.0


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_all():
    """Load TE (encode, free), diffusion model, precompute caches."""
    from futudiffu.text_encoder import create_tokenizer, encode_prompt, load_text_encoder
    from futudiffu.diffusion_model import (
        create_diffusion_model, _detect_cap_feat_dim, _detect_n_layers,
        _detect_qk_norm, _strip_diffusion_prefix,
    )
    from futudiffu.fp8 import replace_linear_with_fp8
    from safetensors.torch import load_file

    print("[load] Text encoder...")
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

    print("[load] Diffusion model (FP8)...")
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

    print(f"[load] Done. dim={model.dim}, {n_layers} layers, "
          f"sigmas [{sigmas[0]:.4f}..{sigmas[-1]:.4f}]")
    return model, pos_cond, neg_cond, sigmas, rope_cache, num_tokens


# ---------------------------------------------------------------------------
# Phase 1: Generate paired training data
# ---------------------------------------------------------------------------

def generate_data(model, pos_cond, neg_cond, sigmas, rope_cache, num_tokens):
    from futudiffu.attention import set_attention_backend
    from futudiffu.sage_attention import configure_sage
    from futudiffu.sampling import (
        const_noise_scaling, const_inverse_noise_scaling,
        const_calculate_denoised, sample_euler,
    )

    log_section("Phase 1: Generating Paired Training Data")
    data_path = Path(DATA_DIR)
    data_path.mkdir(parents=True, exist_ok=True)

    seeds = [BASE_SEED + i for i in range(N_SEEDS)]
    print(f"  Seeds: {seeds}")
    print(f"  Checkpoint steps: {CKPT_STEPS}")

    compiled_model = torch.compile(model, mode="reduce-overhead")
    cond_batch = torch.cat([pos_cond, neg_cond], dim=0)

    def compiled_model_fn(x_in, sigma):
        timestep = sigma * MULTIPLIER
        x_batch = x_in.expand(2, -1, -1, -1)
        t_batch = timestep.expand(2)
        out_batch = compiled_model(x_batch, t_batch, cond_batch,
                                   num_tokens=num_tokens, rope_cache=rope_cache)
        out_cond, out_uncond = out_batch.chunk(2, dim=0)
        d_cond = const_calculate_denoised(sigma, out_cond, x_in)
        d_uncond = const_calculate_denoised(sigma, out_uncond, x_in)
        return d_uncond + (d_cond - d_uncond) * CFG

    backends = [
        ("sdpa", "sdpa", {}),
        ("sage", "sage", dict(smooth_k=True, qk_quant="fp8", pv_quant="bf16")),
    ]

    for backend_name, backend_key, sage_kwargs in backends:
        print(f"\n  --- {backend_name.upper()} renders ---")
        set_attention_backend(backend_key)
        if sage_kwargs:
            configure_sage(**sage_kwargs)
        torch._dynamo.reset()

        print(f"  Warming up torch.compile...")
        warmup_gen = torch.Generator(device=DEVICE).manual_seed(0)
        warmup_noise = torch.randn(1, 16, LATENT_H, LATENT_W,
                                   dtype=DTYPE, generator=warmup_gen, device=DEVICE)
        warmup_x = const_noise_scaling(sigmas[0], warmup_noise,
                                       torch.zeros_like(warmup_noise))
        with torch.inference_mode():
            sample_euler(compiled_model_fn, warmup_x, sigmas)
        del warmup_noise, warmup_x
        torch.cuda.empty_cache()
        print(f"  Warmup done.")

        for seed in seeds:
            t0 = time.time()
            save_path = data_path / f"seed_{seed}_{backend_name}.pt"
            if save_path.exists():
                print(f"  seed {seed}: already exists, skipping")
                continue

            gen = torch.Generator(device=DEVICE).manual_seed(seed)
            noise = torch.randn(1, 16, LATENT_H, LATENT_W,
                                dtype=DTYPE, generator=gen, device=DEVICE)
            latent = torch.zeros(1, 16, LATENT_H, LATENT_W,
                                 device=DEVICE, dtype=DTYPE)
            x = const_noise_scaling(sigmas[0], noise, latent)

            step_latents = {}
            def _callback(info):
                i = info['i']
                if i in CKPT_STEPS:
                    step_latents[i] = info['x'].detach().cpu().clone()

            with torch.inference_mode():
                x_final = sample_euler(compiled_model_fn, x, sigmas, callback=_callback)
                x_final = const_inverse_noise_scaling(sigmas[-1], x_final)

            payload = {
                "seed": seed, "backend": backend_name,
                "final": x_final.detach().cpu(),
                "noise": noise.detach().cpu(),
            }
            for step_idx, lat in step_latents.items():
                payload[f"step_{step_idx:02d}"] = lat

            torch.save(payload, str(save_path))
            dt = time.time() - t0
            print(f"  seed {seed}: {dt:.1f}s, saved {save_path.name}")
            del noise, latent, x, x_final, step_latents
            torch.cuda.empty_cache()

    torch._dynamo.reset()
    print(f"\nData generation complete.")


# ---------------------------------------------------------------------------
# Phase 2: Train R_theta
# ---------------------------------------------------------------------------

def train_btrm(model, sigmas, rope_cache, num_tokens, pos_cond):
    """Train R_theta = frozen backbone + LoRA("rtheta", layers 28-29) + ScoreUnembedder."""
    from futudiffu.attention import set_attention_backend
    from futudiffu.btrm import ScoreUnembedder, bradley_terry_loss, logsquare_regularizer
    from futudiffu.lora import inject_lora, get_lora_params, lora_state_dict

    log_section("Phase 2: R_theta Training (LoRA layers 28-29 + ScoreUnembedder)")
    set_attention_backend("sdpa")

    injected = inject_lora(model, name="rtheta", rank=LORA_RANK, alpha=LORA_ALPHA,
                           layer_indices={28, 29})
    rtheta_params = list(get_lora_params(model, adapter_name="rtheta"))
    n_lora = sum(p.numel() for p in rtheta_params)
    print(f"  R_theta LoRA: {len(injected)} adapters on layers 28-29, "
          f"{n_lora:,} params ({n_lora*2/1024**2:.1f} MB)")

    btrm = ScoreUnembedder(
        hidden_dim=model.dim,
        head_names=("bit_quality", "step_quality"),
        logit_cap=10.0,
    ).to(device=DEVICE, dtype=DTYPE)
    n_head = sum(p.numel() for p in btrm.parameters())
    print(f"  R_theta head: {n_head:,} params")

    all_params = rtheta_params + list(btrm.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=BTRM_LR)

    data_path = Path(DATA_DIR)
    seeds = [BASE_SEED + i for i in range(N_SEEDS)]
    pairs = [(seed, si, sigmas[si].item()) for seed in seeds for si in CKPT_STEPS]
    print(f"  {len(pairs)} pairs x {BTRM_EPOCHS} epochs = {len(pairs)*BTRM_EPOCHS} steps")

    log = []
    global_step = 0

    for epoch in range(BTRM_EPOCHS):
        import random
        random.shuffle(pairs)
        epoch_losses = []

        for seed, step_idx, sigma_val in pairs:
            t0 = time.time()

            sdpa_data = torch.load(
                str(data_path / f"seed_{seed}_sdpa.pt"), weights_only=True)
            sage_data = torch.load(
                str(data_path / f"seed_{seed}_sage.pt"), weights_only=True)

            key = f"step_{step_idx:02d}"
            x_pos = sdpa_data[key].to(device=DEVICE, dtype=DTYPE)
            x_neg = sage_data[key].to(device=DEVICE, dtype=DTYPE)

            sigma = torch.tensor([sigma_val], device=DEVICE, dtype=DTYPE)
            timestep = sigma * MULTIPLIER

            _, hidden_pos = forward_checkpointed(
                model, x_pos, timestep, pos_cond,
                num_tokens=num_tokens, rope_cache=rope_cache)
            _, hidden_neg = forward_checkpointed(
                model, x_neg, timestep, pos_cond,
                num_tokens=num_tokens, rope_cache=rope_cache)

            with torch.no_grad():
                h_cos = F.cosine_similarity(
                    hidden_pos.flatten().unsqueeze(0),
                    hidden_neg.flatten().unsqueeze(0)).item()
                x_cos = F.cosine_similarity(
                    x_pos.flatten().unsqueeze(0),
                    x_neg.flatten().unsqueeze(0)).item()

            pos_scores = btrm(hidden_pos)
            neg_scores = btrm(hidden_neg)
            bt_loss = bradley_terry_loss(pos_scores[:, 0], neg_scores[:, 0])
            logsq = logsquare_regularizer(pos_scores[:, 0])
            loss = bt_loss + BTRM_LOGSQ_WEIGHT * logsq

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            dt = time.time() - t0
            margin = (pos_scores[:, 0] - neg_scores[:, 0]).detach().item()
            epoch_losses.append(loss.item())

            entry = dict(
                epoch=epoch, step=global_step, seed=seed, step_idx=step_idx,
                sigma=sigma_val, loss=loss.item(), bt_loss=bt_loss.item(),
                margin=margin, x_cos=x_cos, h_cos=h_cos, time=dt,
            )
            log.append(entry)

            if global_step % 5 == 0 or step_idx == CKPT_STEPS[-1]:
                print(
                    f"  ep{epoch} step{global_step:3d} | "
                    f"seed={seed} t={step_idx:2d} sigma={sigma_val:.4f} | "
                    f"loss={loss.item():.4f} bt={bt_loss.item():.4f} "
                    f"margin={margin:+.4f} | "
                    f"x_cos={x_cos:.6f} h_cos={h_cos:.6f} | {dt:.1f}s"
                )

            global_step += 1
            del x_pos, x_neg, hidden_pos, hidden_neg, sdpa_data, sage_data
            torch.cuda.empty_cache()

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"  --- epoch {epoch} avg_loss={avg_loss:.4f} ---")

    print(f"\nHidden-state & latent divergence by step (last epoch):")
    last_epoch = [e for e in log if e['epoch'] == BTRM_EPOCHS - 1]
    for step_idx in CKPT_STEPS:
        entries = [e for e in last_epoch if e['step_idx'] == step_idx]
        if entries:
            avg_x = sum(e['x_cos'] for e in entries) / len(entries)
            avg_h = sum(e['h_cos'] for e in entries) / len(entries)
            avg_m = sum(e['margin'] for e in entries) / len(entries)
            print(f"  step {step_idx:2d}: x_cos={avg_x:.6f} h_cos={avg_h:.6f} "
                  f"margin={avg_m:+.4f}")

    rtheta_sd = lora_state_dict(model, adapter_name="rtheta")
    rtheta_sd.update({f"btrm.{k}": v.cpu() for k, v in btrm.state_dict().items()})
    rtheta_path = os.path.join(DATA_DIR, "rtheta_weights.pt")
    torch.save(rtheta_sd, rtheta_path)
    print(f"\n  Saved R_theta ({len(rtheta_sd)} tensors) to {rtheta_path}")

    return btrm, log


# ---------------------------------------------------------------------------
# Phase 3: Policy optimization
# ---------------------------------------------------------------------------

def train_policy(model, btrm, pos_cond, sigmas, rope_cache, num_tokens):
    """Train p_theta with stacked "ptheta" adapter + frozen "rtheta" reward."""
    from futudiffu.lora import (
        LoRALinear, inject_lora, get_lora_params,
        lora_state_dict, set_lora_scale, freeze_adapter,
    )
    from futudiffu.attention import set_attention_backend
    from futudiffu.sage_attention import configure_sage

    log_section("Phase 3: p_theta Policy Optimization (stacked LoRA + BTRM reward)")

    btrm.eval()
    for p in btrm.parameters():
        p.requires_grad_(False)

    n_frozen = freeze_adapter(model, "rtheta")
    set_lora_scale(model, torch.tensor([0.0], device=DEVICE, dtype=DTYPE),
                   adapter_name="rtheta")
    print(f"  rtheta: {n_frozen} adapters frozen + zeroed (scale=0)")

    injected = inject_lora(model, name="ptheta", rank=LORA_RANK, alpha=LORA_ALPHA)
    ptheta_params = list(get_lora_params(model, adapter_name="ptheta"))
    n_trainable = sum(p.numel() for p in ptheta_params)
    n_main = sum(1 for n in injected if n.startswith("layers."))
    print(f"  ptheta: {len(injected)} adapters ({n_main} main, "
          f"{len(injected)-n_main} refiner), "
          f"{n_trainable:,} params ({n_trainable*2/1024**2:.1f} MB)")

    n_stacked = 0
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALinear) and len(mod.adapters) > 1:
            n_stacked += 1
    print(f"  Stacked (rtheta+ptheta) adapters: {n_stacked}")

    sample_params = OrderedDict()
    for name, mod in model.named_modules():
        if not isinstance(mod, LoRALinear):
            continue
        if "ptheta" not in mod.adapters:
            continue
        if any(p in name for p in [
            "layers.0.attention.out", "layers.15.attention.out",
            "layers.29.attention.out", "layers.29.attention.qkv",
        ]):
            a = mod.adapters["ptheta"]
            sample_params[f"{name}.ptheta.A"] = a.lora_A
            sample_params[f"{name}.ptheta.B"] = a.lora_B

    optimizer = torch.optim.AdamW(ptheta_params, lr=POLICY_LR)

    set_attention_backend("sage")
    configure_sage(smooth_k=True, qk_quant="fp8", pv_quant="bf16")

    n_sigmas = len(sigmas) - 1
    sigma_indices = [int(i * (n_sigmas - 1) / (POLICY_STEPS - 1))
                     for i in range(POLICY_STEPS)]
    print(f"  Probe sigmas: {[(i, sigmas[i].item()) for i in sigma_indices]}")

    log = []
    for step in range(POLICY_STEPS):
        t0 = time.time()
        prev = {n: p.data.clone() for n, p in sample_params.items()}

        sigma_idx = sigma_indices[step]
        sigma = sigmas[sigma_idx]
        timestep = (sigma * MULTIPLIER).unsqueeze(0)

        noise = torch.randn(1, 16, LATENT_H, LATENT_W, dtype=DTYPE, device=DEVICE)
        x_t = sigma * noise

        _, last_hidden = forward_checkpointed(
            model, x_t, timestep, pos_cond,
            num_tokens=num_tokens, rope_cache=rope_cache)

        scores = btrm(last_hidden)
        reward_bq = scores[:, 0].mean()
        loss = -reward_bq

        optimizer.zero_grad()
        loss.backward()
        mem = torch.cuda.max_memory_allocated() / 1024**3
        grad_norm = torch.nn.utils.clip_grad_norm_(ptheta_params, GRAD_CLIP)
        optimizer.step()

        dt = time.time() - t0
        n_grad = sum(1 for p in ptheta_params
                     if p.grad is not None and p.grad.abs().sum() > 0)

        entry = dict(step=step, sigma=sigma.item(), reward=reward_bq.item(),
                     loss=loss.item(), grad_norm=grad_norm.item()
                     if isinstance(grad_norm, torch.Tensor) else grad_norm,
                     n_grad=n_grad, n_total=len(ptheta_params), mem=mem, time=dt)
        log.append(entry)

        print(f"  step {step} | sigma={sigma.item():.4f} | "
              f"reward={reward_bq.item():+.4f} | grad_norm={entry['grad_norm']:.3e} | "
              f"grads: {n_grad}/{len(ptheta_params)} | "
              f"mem={mem:.1f}GB | {dt:.1f}s")

        for pname, param in sample_params.items():
            delta = (param.data - prev[pname]).norm().item()
            gn = param.grad.norm().item() if param.grad is not None else 0.0
            flag = " [zero]" if gn == 0 else ""
            print(f"    {pname:50s} grad={gn:.3e} delta={delta:.3e}{flag}")

        del noise, x_t, last_hidden, scores
        torch.cuda.empty_cache()

    print(f"\nReward trajectory: {[f'{e['reward']:+.4f}' for e in log]}")

    sd = lora_state_dict(model, adapter_name="ptheta")
    save_path = r"F:\dox\repos\ai\futudiffu\smoke_test_policy_v2.safetensors"
    from safetensors.torch import save_file
    save_file({k: v.cpu() for k, v in sd.items()}, save_path)
    print(f"  Saved {len(sd)} ptheta LoRA tensors to {save_path}")

    return log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_total = time.time()
    log_section("BTRM v2: Paired Trajectory Training (horizontal multi-LoRA)")

    model, pos_cond, neg_cond, sigmas, rope_cache, num_tokens = load_all()

    # Phase 1
    data_path = Path(DATA_DIR)
    all_exist = all(
        (data_path / f"seed_{BASE_SEED+i}_{b}.pt").exists()
        for i in range(N_SEEDS) for b in ("sdpa", "sage")
    )
    if all_exist:
        print(f"\nAll {N_SEEDS*2} data files exist, skipping Phase 1.")
    else:
        generate_data(model, pos_cond, neg_cond, sigmas, rope_cache, num_tokens)

    # Phase 2
    btrm, btrm_log = train_btrm(model, sigmas, rope_cache, num_tokens, pos_cond)

    # Phase 3
    policy_log = train_policy(model, btrm, pos_cond, sigmas, rope_cache, num_tokens)

    # Summary
    log_section("Final Summary")
    all_ok = True
    for e in btrm_log:
        if math.isnan(e['loss']):
            all_ok = False
    for e in policy_log:
        if math.isnan(e['loss']):
            all_ok = False

    elapsed = time.time() - t_total
    mem = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Total time: {elapsed:.1f}s")
    print(f"Peak GPU memory: {mem:.1f} GB")
    print(f"Result: {'PASS (no NaN)' if all_ok else 'FAIL (NaN detected)'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

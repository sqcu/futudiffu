"""Generate laser shark reference image with policy LoRA on vs off.

Connects to running server, generates the golden reference prompt with:
  1. ptheta active (scale=1.0) — policy-modified image
  2. ptheta off (scale=0.0) — baseline reference image
  3. Also saves the ptheta LoRA weights to safetensors

Outputs to bench_renders/policy_on.png, bench_renders/policy_off.png,
and bench_renders/ptheta_lora.safetensors.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from futudiffu.client import InferenceClient

GOLDEN_PROMPT = (
    'ahem.\n*ting ting ting ting ting*\n'
    'the query model for this is a LARGE LANGUAGE MODEL, specifically '
    'QWEN-3-4B, a GENERAL PURPOSE SEMANTIC PARSER which is able to WRITE '
    'SENTENCES AT A TIME when they are participating in dialogue. however, '
    'in this situation, they are being used as a hidden state generator to '
    'steer an *image generation model*, z-image.\n\n'
    'qwen-3-4b, draw me an "enormous laser shark for the sega saturn".'
)

SEED = 91849188298864
STEPS = 30
CFG = 4.0
WIDTH = 1280
HEIGHT = 832


def generate_and_decode(client, pos_cond, neg_cond, label, seed=SEED):
    """Generate a full 30-step rollout and decode to PIL image."""
    print(f"\n  Generating ({label})...")
    t0 = time.time()
    result = client.sample_trajectory(
        pos_cond=pos_cond,
        neg_cond=neg_cond,
        seed=seed,
        n_steps=STEPS,
        cfg=CFG,
        width=WIDTH,
        height=HEIGHT,
        attention_backend="sdpa",
        save_steps=[],  # only need final
    )
    dt_gen = time.time() - t0
    print(f"    Generation: {dt_gen:.1f}s")

    final_latent = result["final"]
    print(f"    Decoding...")
    t0 = time.time()
    pixels = client.vae_decode(final_latent)
    dt_dec = time.time() - t0
    print(f"    Decode: {dt_dec:.1f}s")

    # pixels: (1, C, H, W) float tensor -> (H, W, C) uint8 numpy
    img_tensor = pixels[0].clamp(0, 1)
    img_np = (img_tensor.permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(img_np)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--output-dir", default="bench_renders")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    client = InferenceClient(f"tcp://localhost:{args.port}")
    status = client.status()
    print(f"Server status: {status}")

    # Encode prompts
    print("Encoding prompts...")
    pos_cond = client.encode_prompt(GOLDEN_PROMPT)
    neg_cond = client.encode_prompt("")
    print(f"  pos: {pos_cond.shape}, neg: {neg_cond.shape}")

    # --- Generate with ptheta ON ---
    client.set_adapter_config("ptheta", scale=1.0)
    img_on = generate_and_decode(client, pos_cond, neg_cond, "ptheta ON")
    path_on = os.path.join(args.output_dir, "policy_on.png")
    img_on.save(path_on)
    print(f"  Saved: {path_on}")

    # --- Generate with ptheta OFF ---
    client.set_adapter_config("ptheta", scale=0.0)
    img_off = generate_and_decode(client, pos_cond, neg_cond, "ptheta OFF")
    path_off = os.path.join(args.output_dir, "policy_off.png")
    img_off.save(path_off)
    print(f"  Saved: {path_off}")

    # --- Also score both with BTRM head if available ---
    # (the BTRM head lives on the client side in the E2E test,
    #  not on the server, so we'd need to reconstruct it. Skip for now.)

    # --- Save ptheta LoRA weights ---
    print("\nSaving ptheta LoRA weights...")
    ptheta_sd = client.get_lora_state_dict("ptheta")
    lora_path = os.path.join(args.output_dir, "ptheta_lora.safetensors")
    try:
        from safetensors.torch import save_file
        save_file({k: v.cpu().contiguous() for k, v in ptheta_sd.items()}, lora_path)
        print(f"  Saved {len(ptheta_sd)} tensors to {lora_path}")
    except ImportError:
        # Fallback to torch.save
        lora_path = os.path.join(args.output_dir, "ptheta_lora.pt")
        torch.save(ptheta_sd, lora_path)
        print(f"  Saved {len(ptheta_sd)} tensors to {lora_path} (torch format)")

    # Restore ptheta scale
    client.set_adapter_config("ptheta", scale=1.0)

    print("\nDone. Compare:")
    print(f"  {path_off}  (baseline, no LoRA)")
    print(f"  {path_on}   (policy LoRA active)")


if __name__ == "__main__":
    main()

"""Generate policy evaluation dataset: trajectories with ptheta active.

Records full trajectories (intermediate checkpoints + finals) for both
t2i and i2i, enabling ex post facto comparison between policy and
reference model outputs on reproducible seeds.

Schedule:
  - 12 t2i prompts (first 12 from BTRM training templates), 1 seed each
  - 11 i2i images x 1 denoise strength (0.85), rendering finals for all

All trajectories use ptheta at scale=1.0 via the server.  A parallel
"reference" run with scale=0.0 can be done afterward for paired comparison.

Usage:
  # Terminal 1: server
  .venv/Scripts/python.exe -m futudiffu.server --port 5555 ...

  # Terminal 2: generate (injects + loads ptheta weights automatically)
  .venv/Scripts/python.exe generate_policy_eval.py --port 5555 \
      --lora-weights smoke_test_policy_v2.safetensors \
      --output-dir policy_eval_dataset
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from futudiffu.btrm_dataset import I2I_IMAGES, PROMPT_TEMPLATES
from futudiffu.client import InferenceClient


# ---------------------------------------------------------------------------
# I2I image loading (from generate_btrm_dataset.py)
# ---------------------------------------------------------------------------

I2I_DIR = Path(__file__).parent / "i2i_off_policies"


def load_i2i_image(image_path: Path, multiple: int = 16):
    """Load image at native resolution, center-crop to 16-multiple."""
    img = Image.open(str(image_path)).convert("RGB")
    src_w, src_h = img.size
    crop_w = (src_w // multiple) * multiple
    crop_h = (src_h // multiple) * multiple
    if crop_w != src_w or crop_h != src_h:
        left = (src_w - crop_w) // 2
        top = (src_h - crop_h) // 2
        img = img.crop((left, top, left + crop_w, top + crop_h))
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(dtype=torch.bfloat16), crop_w, crop_h


def decode_and_save(client, latent, path):
    """VAE decode a latent and save as PNG."""
    pixels = client.vae_decode(latent)
    img = pixels[0].clamp(0, 1).permute(1, 2, 0).float().cpu().numpy()
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(str(path))


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

T2I_PROMPT_INDICES = list(range(12))  # first 12 templates
I2I_DENOISE = 0.85
T2I_STEPS = 30
I2I_STEPS = 30
CFG = 4.0
WIDTH = 1280
HEIGHT = 832
SAVE_STEPS = [0, 4, 9, 14, 19, 24, 29]
BASE_SEED = 20260214  # today's date


def generate_t2i(client, prompt_idx, seed, output_dir, pos_cache, neg_cond,
                 policy_scale):
    """Generate one t2i trajectory."""
    prompt = PROMPT_TEMPLATES[prompt_idx]
    if prompt not in pos_cache:
        pos_cache[prompt] = client.encode_prompt(prompt)
    pos_cond = pos_cache[prompt]

    result = client.sample_trajectory(
        pos_cond=pos_cond,
        neg_cond=neg_cond,
        seed=seed,
        n_steps=T2I_STEPS,
        cfg=CFG,
        width=WIDTH,
        height=HEIGHT,
        attention_backend="sdpa",
        save_steps=SAVE_STEPS,
    )

    # Save checkpoints
    traj_dir = output_dir / "latents" / f"traj_{prompt_idx:03d}_s{seed}"
    traj_dir.mkdir(parents=True, exist_ok=True)
    for key, tensor in result.items():
        torch.save(tensor.cpu(), traj_dir / f"{key}.pt")

    # Save meta
    meta = {
        "type": "t2i",
        "prompt_idx": prompt_idx,
        "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
        "seed": seed,
        "n_steps": T2I_STEPS,
        "cfg": CFG,
        "width": WIDTH,
        "height": HEIGHT,
        "policy_scale": policy_scale,
    }
    with open(traj_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Render final
    render_dir = output_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    render_path = render_dir / f"t2i_{prompt_idx:03d}_s{seed}.png"
    decode_and_save(client, result["final"], render_path)

    return meta, render_path


def generate_i2i(client, image_idx, image_info, seed, output_dir, neg_cond,
                 policy_scale):
    """Generate one i2i trajectory."""
    filename = image_info["filename"]
    object_label = image_info["object_label"]
    image_path = I2I_DIR / filename

    if not image_path.exists():
        print(f"    SKIP: {filename} not found")
        return None, None

    # Load and encode source image
    img_tensor, crop_w, crop_h = load_i2i_image(image_path)
    clean_latent = client.vae_encode(img_tensor)

    # Build prompt: object label + style transformation
    prompt = f"{object_label}, rendered with enhanced clarity and detail"
    pos_cond = client.encode_prompt(prompt)

    result = client.sample_trajectory(
        pos_cond=pos_cond,
        neg_cond=neg_cond,
        seed=seed,
        n_steps=I2I_STEPS,
        cfg=CFG,
        width=crop_w,
        height=crop_h,
        attention_backend="sdpa",
        save_steps=SAVE_STEPS,
        denoise=I2I_DENOISE,
        clean_latent=clean_latent,
    )

    # Save checkpoints
    safe_name = filename.replace(" ", "_").rsplit(".", 1)[0]
    traj_dir = output_dir / "latents" / f"i2i_{image_idx:02d}_{safe_name}"
    traj_dir.mkdir(parents=True, exist_ok=True)
    for key, tensor in result.items():
        torch.save(tensor.cpu(), traj_dir / f"{key}.pt")

    meta = {
        "type": "i2i",
        "image_idx": image_idx,
        "image_file": filename,
        "object_label": object_label,
        "prompt": prompt,
        "seed": seed,
        "n_steps": I2I_STEPS,
        "cfg": CFG,
        "width": crop_w,
        "height": crop_h,
        "denoise": I2I_DENOISE,
        "policy_scale": policy_scale,
    }
    with open(traj_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Render final
    render_dir = output_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    render_path = render_dir / f"i2i_{image_idx:02d}_{safe_name}.png"
    decode_and_save(client, result["final"], render_path)

    # Also copy source image for side-by-side comparison
    source_copy = render_dir / f"i2i_{image_idx:02d}_{safe_name}_source.png"
    if not source_copy.exists():
        Image.open(str(image_path)).save(str(source_copy))

    return meta, render_path


def setup_ptheta(client, lora_weights_path, policy_scale):
    """Inject ptheta LoRA and load trained weights from safetensors file."""
    print(f"\nInjecting ptheta LoRA (rank=8, alpha=16.0, all layers)...")
    n = client.inject_lora("ptheta", rank=8, alpha=16.0)
    print(f"  Injected {n} adapters")

    print(f"  Loading weights from {lora_weights_path}")
    sd = load_file(str(lora_weights_path))
    print(f"  {len(sd)} weight tensors loaded from file")
    client.update_lora_weights(sd)
    print(f"  Weights pushed to server")

    print(f"  Setting ptheta scale={policy_scale}")
    client.set_adapter_config("ptheta", scale=policy_scale)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--output-dir", type=str, default="policy_eval_dataset")
    parser.add_argument("--policy-scale", type=float, default=1.0,
                        help="ptheta scale (1.0=policy on, 0.0=reference)")
    parser.add_argument("--lora-weights", type=str, default=None,
                        help="Path to ptheta safetensors weights file")
    parser.add_argument("--skip-t2i", action="store_true")
    parser.add_argument("--skip-i2i", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = InferenceClient(f"tcp://localhost:{args.port}")
    status = client.status()
    print(f"Server status: {status}")

    # Inject ptheta and load weights, or just set scale if already present
    if args.lora_weights:
        setup_ptheta(client, args.lora_weights, args.policy_scale)
    else:
        print(f"\nSetting ptheta scale={args.policy_scale}")
        client.set_adapter_config("ptheta", scale=args.policy_scale)

    # Encode negative prompt once
    neg_cond = client.encode_prompt("")
    pos_cache = {}

    manifest = {
        "policy_scale": args.policy_scale,
        "t2i_records": [],
        "i2i_records": [],
        "generation_time": None,
    }

    t_start = time.time()

    # --- T2I ---
    if not args.skip_t2i:
        print(f"\n{'='*60}")
        print(f"  T2I: {len(T2I_PROMPT_INDICES)} prompts, {T2I_STEPS} steps")
        print(f"{'='*60}")

        for i, pidx in enumerate(T2I_PROMPT_INDICES):
            seed = BASE_SEED + pidx
            prompt_preview = PROMPT_TEMPLATES[pidx][:60]
            print(f"\n  [{i+1}/{len(T2I_PROMPT_INDICES)}] prompt {pidx}: "
                  f"{prompt_preview}...")

            t0 = time.time()
            meta, render_path = generate_t2i(
                client, pidx, seed, output_dir, pos_cache, neg_cond,
                args.policy_scale,
            )
            dt = time.time() - t0
            manifest["t2i_records"].append(meta)
            print(f"    {dt:.1f}s -> {render_path}")

    # --- I2I ---
    if not args.skip_i2i:
        print(f"\n{'='*60}")
        print(f"  I2I: {len(I2I_IMAGES)} images, denoise={I2I_DENOISE}")
        print(f"{'='*60}")

        for i, img_info in enumerate(I2I_IMAGES):
            seed = BASE_SEED + 1000 + i
            print(f"\n  [{i+1}/{len(I2I_IMAGES)}] {img_info['filename']}")

            t0 = time.time()
            meta, render_path = generate_i2i(
                client, i, img_info, seed, output_dir, neg_cond,
                args.policy_scale,
            )
            dt = time.time() - t0
            if meta:
                manifest["i2i_records"].append(meta)
                print(f"    {dt:.1f}s -> {render_path}")

    elapsed = time.time() - t_start
    manifest["generation_time"] = elapsed

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Done in {elapsed:.0f}s")
    print(f"  {len(manifest['t2i_records'])} t2i + "
          f"{len(manifest['i2i_records'])} i2i trajectories")
    print(f"  Output: {output_dir}")
    print(f"  Manifest: {manifest_path}")
    print(f"{'='*60}")

    # Suggest reference run
    if args.policy_scale != 0.0:
        ref_dir = str(output_dir).replace("policy", "reference")
        print(f"\nTo generate paired reference (no policy):")
        print(f"  .venv/Scripts/python.exe generate_policy_eval.py "
              f"--policy-scale 0.0 --output-dir {ref_dir}")


if __name__ == "__main__":
    main()

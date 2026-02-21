"""Render trajectory comparison: VAE decode both reference and reproduced latents,
produce per-step ref/repro/diff PNGs, step-strip composites, and a final timeline.

Uses src_ii.vae_utils for VAE loading and decoding instead of inlining.

Outputs saved to: validation_output_ii/traj_000000/renders/
"""

import os
import sys

sys.path.insert(0, r"F:\dox\repos\ai\futudiffu\src")
sys.path.insert(0, r"F:\dox\repos\ai\futudiffu")

import torch
from PIL import Image

from src_ii.vae_utils import load_vae, decode_latent_to_pil
from src_ii.rendering import make_false_color_diff
from src_ii.visualization import render_strip

VAE_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\vae\zimage.safetensors"
REF_DIR = r"F:\dox\repos\ai\futudiffu\btrm_dataset\latents\traj_000000"
REPRO_DIR = r"F:\dox\repos\ai\futudiffu\validation_output_ii\traj_000000"
OUT_DIR = r"F:\dox\repos\ai\futudiffu\validation_output_ii\traj_000000\renders"

STEPS = ["step_00", "step_04", "step_09", "step_14", "step_19", "step_24", "step_29", "final"]

DEVICE = torch.device("cuda")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    vae = load_vae(VAE_PATH, device=DEVICE, dtype=torch.bfloat16)

    decoded_refs = {}
    decoded_repros = {}
    decoded_diffs = {}

    for step in STEPS:
        ref_pt = os.path.join(REF_DIR, f"{step}.pt")
        repro_pt = os.path.join(REPRO_DIR, f"{step}.pt")

        if not os.path.exists(ref_pt):
            print(f"[SKIP] Missing reference: {ref_pt}")
            continue
        if not os.path.exists(repro_pt):
            print(f"[SKIP] Missing reproduced: {repro_pt}")
            continue

        print(f"[{step}] Decoding reference ...")
        ref_latent = torch.load(ref_pt, map_location=DEVICE, weights_only=True)
        ref_img = decode_latent_to_pil(vae, ref_latent, device=DEVICE, dtype=torch.bfloat16)

        print(f"[{step}] Decoding reproduced ...")
        repro_latent = torch.load(repro_pt, map_location=DEVICE, weights_only=True)
        repro_img = decode_latent_to_pil(vae, repro_latent, device=DEVICE, dtype=torch.bfloat16)

        print(f"[{step}] Computing diff ...")
        diff_img = make_false_color_diff(ref_img, repro_img, scale=10.0)

        suffix = "final" if step == "final" else step

        ref_out = os.path.join(OUT_DIR, f"ref_{suffix}.png")
        repro_out = os.path.join(OUT_DIR, f"repro_{suffix}.png")
        diff_out = os.path.join(OUT_DIR, f"diff_{suffix}.png")

        ref_img.save(ref_out)
        repro_img.save(repro_out)
        diff_img.save(diff_out)

        # Step strip using extracted render_strip
        strip = render_strip(
            [("ref", ref_img), ("repro", repro_img), ("diff", diff_img)],
            max_width=ref_img.width * 3,
        )
        strip_out = os.path.join(OUT_DIR, f"strip_{suffix}.png")
        strip.save(strip_out)
        print(f"  Saved: {strip_out}")

        decoded_refs[step] = ref_img
        decoded_repros[step] = repro_img
        decoded_diffs[step] = diff_img

    if "final" in decoded_refs:
        timeline = render_strip(
            [("ref_final", decoded_refs["final"]),
             ("repro_final", decoded_repros["final"]),
             ("diff_final", decoded_diffs["final"])],
            max_width=decoded_refs["final"].width * 3,
        )
        timeline_out = os.path.join(OUT_DIR, "timeline_final.png")
        timeline.save(timeline_out)
        print(f"[timeline] Saved: {timeline_out}")

    print("\nAll renders complete.")
    print(f"Output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()

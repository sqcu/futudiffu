"""QAT (Quantization-Aware Training) CLI for Z-Image NextDiT LoRA adapters.

Thin CLI wrapper around trainer.py. Supports two modes:
  drgrpo: DRGRPO policy optimization with BTRM reward + reference anchoring.
  legacy: Original direct/reinforce modes (retained for compatibility).

Usage:
  python -m futudiffu.train_qat --diffusion-model PATH --text-encoder PATH \\
      --mode drgrpo --policy-iterations 20 --lr 1e-4
"""

import sys
import time
from dataclasses import dataclass

import torch


# ---------------------------------------------------------------------------
# Config (maps CLI args to TrainConfig)
# ---------------------------------------------------------------------------

LASER_SHARK_PROMPT = (
    'ahem.\n*ting ting ting ting ting*\n'
    'the query model for this is a LARGE LANGUAGE MODEL, specifically QWEN-3-4B, '
    'a GENERAL PURPOSE SEMANTIC PARSER which is able to WRITE SENTENCES AT A TIME '
    'when they are participating in dialogue. however, in this situation, they are '
    'being used as a hidden state generator to steer an *image generation model*, '
    'z-image.\n\nqwen-3-4b, draw me an "enormous laser shark for the sega saturn".'
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_qat(args) -> int:
    """Run training via trainer.py. Returns 0 on success, 1 on failure."""
    from .trainer import TrainConfig, setup_training, train_loop

    config = TrainConfig(
        diffusion_model_path=args.diffusion_model,
        text_encoder_path=args.text_encoder,
        vae_path=args.vae,
        tokenizer_path=args.tokenizer,
        width=args.width,
        height=args.height,
        steps=args.steps,
        cfg=args.cfg,
        sampling_shift=getattr(args, 'sampling_shift', 1.0),
        multiplier=getattr(args, 'multiplier', 1.0),
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        btrm_lr=getattr(args, 'btrm_lr', 1e-3),
        btrm_epochs=getattr(args, 'btrm_epochs', 3),
        policy_lr=args.lr,
        policy_iterations=args.num_iterations,
        grad_clip=args.grad_clip,
        group_size=args.group_size,
        sparse_steps=args.sparse_steps,
        s_churn=args.s_churn,
        clip_low=getattr(args, 'clip_low', 0.2),
        clip_high=getattr(args, 'clip_high', 0.28),
        lambda_ent=getattr(args, 'lambda_ent', 0.01),
        lambda_anchor=getattr(args, 'lambda_anchor', 1e-4),
        sage_smooth_k=args.sage_smooth_k,
        sage_qk_quant=args.sage_qk_quant,
        sage_pv_quant=args.sage_pv_quant,
        device=args.device,
        dtype=args.dtype,
        fp8_diffusion=args.fp8_diffusion,
        fp8_block_size=args.fp8_block_size,
        save_dir=getattr(args, 'save_dir', 'training_output'),
    )

    t_total = time.time()

    print("=" * 70)
    print("futudiffu QAT Training (unified trainer)")
    print("=" * 70)
    print(f"  Iterations: {config.policy_iterations}")
    print(f"  LR: {config.policy_lr}")
    print(f"  LoRA rank={config.lora_rank}, alpha={config.lora_alpha}")
    print(f"  Steps: {config.steps}, CFG: {config.cfg}")
    print(f"  Sage: qk={config.sage_qk_quant}, pv={config.sage_pv_quant}")
    print(f"  Resolution: {config.width}x{config.height}")
    print(f"  FP8 diffusion: {config.fp8_diffusion}")
    print(f"  DRGRPO: clip=[{config.clip_low}, {config.clip_high}], "
          f"lambda_ent={config.lambda_ent}, lambda_anchor={config.lambda_anchor}")
    print()

    state = setup_training(config)
    result = train_loop(state, config)

    elapsed = time.time() - t_total
    print(f"\nTotal training time: {elapsed:.1f}s")
    return 0 if result["ok"] else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="futudiffu QAT: train LoRA adapters with DRGRPO + BTRM"
    )
    parser.add_argument("--diffusion-model", required=True,
                        help="Path to diffusion model safetensors")
    parser.add_argument("--text-encoder", required=True,
                        help="Path to text encoder safetensors")
    parser.add_argument("--vae", required=True,
                        help="Path to VAE safetensors (unused, for interface compat)")
    parser.add_argument("--tokenizer", default=None,
                        help="Path to tokenizer directory")
    parser.add_argument("--num-iterations", type=int, default=20,
                        help="Policy optimization iterations")
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
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--sparse-steps", type=int, default=5)
    parser.add_argument("--s-churn", type=float, default=0.0)
    parser.add_argument("--btrm-lr", type=float, default=1e-3)
    parser.add_argument("--btrm-epochs", type=int, default=3)
    parser.add_argument("--clip-low", type=float, default=0.2)
    parser.add_argument("--clip-high", type=float, default=0.28)
    parser.add_argument("--lambda-ent", type=float, default=0.01)
    parser.add_argument("--lambda-anchor", type=float, default=1e-4)
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
    parser.add_argument("--save-dir", default="training_output",
                        help="Output directory for weights")

    args = parser.parse_args()
    sys.exit(train_qat(args))


if __name__ == "__main__":
    main()

"""Text encoding lifecycle for inference pipelines.

This module wraps the Qwen TE load/encode/free lifecycle into a single
importable function. Callers never touch the TE object directly.

Import constraint: `futudiffu.text_encoder` lives in the frozen src/ library
(src/futudiffu/text_encoder.py). This is the one permitted import from
src/futudiffu/ in src_ii/ code, because text encoding has no algorithmic
content that needs rewriting -- it is pure weight loading and tokenization
glue. The import is deferred inside the function body to avoid making the
frozen library a module-level dependency of src_ii/.
"""

from __future__ import annotations

import gc
import time

import torch


def encode_prompts(
    prompts: list[tuple[str, str]],  # [(slug, prompt_text), ...]
    te_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Encode prompts via Qwen TE. Returns CPU tensors. Frees TE after.

    Owns the entire TE lifecycle: load, encode all prompts, delete, flush
    CUDA cache. The caller receives a dict of CPU tensors and has no
    further responsibility for the TE.

    Args:
        prompts: List of (slug, prompt_text) pairs. The slug becomes the
            dict key in the returned mapping.
        te_path: Filesystem path to the Qwen text encoder safetensors file.
            Pass a Windows path (e.g. r"F:\\...\\qwen_3_4b.safetensors")
            when running under the WSL2/Windows venv.
        device: Target CUDA device for TE computation.
        dtype: Compute dtype for the TE (typically torch.bfloat16).

    Returns:
        Dict mapping slug -> CPU conditioning tensor. Shape per slug is
        (1, cap_len, cap_feat_dim) where cap_len varies by prompt length.
    """
    # Deferred import: futudiffu.text_encoder is in the frozen src/ library.
    # Only imported here, never at module level, to keep src_ii/ decoupled.
    from futudiffu.text_encoder import (
        create_tokenizer,
        load_text_encoder,
        encode_prompt,
    )

    print("\n" + "=" * 60, flush=True)
    print("  TEXT ENCODING", flush=True)
    print("=" * 60, flush=True)

    t0 = time.perf_counter()

    tokenizer = create_tokenizer()
    te = load_text_encoder(te_path, device=device, dtype=dtype)
    print(
        f"  TE loaded: {time.perf_counter() - t0:.1f}s  "
        f"VRAM {torch.cuda.memory_allocated() / 1e9:.2f} GB",
        flush=True,
    )

    conds: dict[str, torch.Tensor] = {}
    for slug, prompt_text in prompts:
        t_enc = time.perf_counter()
        cond = encode_prompt(te, tokenizer, prompt_text, device=device)
        cond_cpu = cond.cpu()
        conds[slug] = cond_cpu
        print(
            f"  '{slug}': shape {tuple(cond_cpu.shape)}  "
            f"cap_len={cond_cpu.shape[1]}  "
            f"({time.perf_counter() - t_enc:.2f}s)",
            flush=True,
        )

    del te, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(
        f"  TE freed. VRAM {torch.cuda.memory_allocated() / 1e9:.2f} GB  "
        f"total {time.perf_counter() - t0:.1f}s",
        flush=True,
    )

    return conds

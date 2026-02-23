"""Diagnose block mask alignment for packed multi-image forward.

Checks whether segment boundaries in packed sequences align with
SageAttention block sizes (BLOCK_M=128, BLOCK_N=64).

If misaligned, Q blocks straddle two images → second image gets
corrupted attention (attends to wrong image + can't attend to own).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from src_ii.zimage_model import load_zimage_rlaif
from src_ii.forward_packed import prepare_packed_forward
from src_ii.block_mask import BLOCK_M, BLOCK_N

FP8_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\diffusion_models\z_image_fp8_blockwise.safetensors"
TE_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\text_encoders\qwen_3_4b.safetensors"

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16


def main():
    print("=== Block Alignment Diagnostic ===")
    print(f"BLOCK_M (Q tiles) = {BLOCK_M}")
    print(f"BLOCK_N (KV tiles) = {BLOCK_N}")
    print()

    print("Loading model...")
    model = load_zimage_rlaif(FP8_PATH, device=DEVICE, dtype=DTYPE)

    print("Encoding text...")
    from futudiffu.text_encoder import load_text_encoder, encode_text
    te_model, te_tokenizer = load_text_encoder(TE_PATH, device=DEVICE)
    prompt = 'qwen-3-4b, draw me an "enormous laser shark for the sega saturn".'
    pos_cond, cap_len = encode_text(te_model, te_tokenizer, prompt, device=DEVICE)
    neg_cond, neg_cap_len = encode_text(te_model, te_tokenizer, "", device=DEVICE)
    del te_model
    torch.cuda.empty_cache()

    pos_cond = pos_cond.to(dtype=DTYPE)
    neg_cond = neg_cond.to(dtype=DTYPE)
    context = torch.cat([pos_cond, neg_cond], dim=0)  # (2, seq, dim)

    test_cases = [
        [(512, 512), (640, 384)],    # mixed res
        [(512, 512), (512, 512)],    # same res square
        [(640, 384), (640, 384)],    # same res landscape
        [(1024, 1024)],              # single image
        [(256, 256), (320, 320), (512, 512)],  # K=3
    ]

    for resolutions in test_cases:
        K = len(resolutions)
        print(f"\n--- K={K}, resolutions={resolutions} ---")

        patch_size = model.patch_size  # 2
        img_sizes = []
        for w, h in resolutions:
            lat_h, lat_w = h // 8, w // 8
            padded_h = lat_h + (-lat_h % patch_size)
            padded_w = lat_w + (-lat_w % patch_size)
            img_sizes.append((padded_h, padded_w))

        context_list = [context] * K
        cap_lens = [cap_len] * K

        with torch.no_grad():
            from futudiffu.attention import set_attention_backend; set_attention_backend("sdpa")
            state = prepare_packed_forward(
                model, context_list, img_sizes, cap_lens,
                device=DEVICE,
            )

        packing_info = state['packing_info']
        block_mask = state['block_mask']

        print(f"  total_len = {packing_info.total_len}")
        print(f"  pad_tokens_multiple = {model.pad_tokens_multiple}")

        cumulative_offset = 0
        for i, (text_start, text_len, img_start, img_len) in enumerate(packing_info.segments):
            seg_len = text_len + img_len
            cumulative_offset += seg_len
            q_aligned = cumulative_offset % BLOCK_M == 0
            kv_aligned = cumulative_offset % BLOCK_N == 0

            print(f"  Image {i}:")
            print(f"    text_start={text_start}, text_len={text_len}")
            print(f"    img_start={img_start}, img_len={img_len}")
            print(f"    seg_len={seg_len}")
            print(f"    cumulative_offset={cumulative_offset}")
            print(f"    Q-block aligned (mod {BLOCK_M}): {q_aligned}  (remainder={cumulative_offset % BLOCK_M})")
            print(f"    KV-block aligned (mod {BLOCK_N}): {kv_aligned}  (remainder={cumulative_offset % BLOCK_N})")
            if not q_aligned:
                block_start = (cumulative_offset // BLOCK_M) * BLOCK_M
                straddle_from_prev = cumulative_offset - block_start
                straddle_from_next = BLOCK_M - straddle_from_prev
                print(f"    *** MISALIGNED: Q block straddles boundary!")
                print(f"        {straddle_from_prev} tokens from img {i}, {straddle_from_next} tokens from img {i+1}")
                print(f"        {straddle_from_next} tokens of img {i+1} attend to img {i}'s KV!")

        print(f"\n  Block mask shape: {block_mask.shape}")
        if block_mask.shape[0] <= 30:  # Don't print huge masks
            mask_str = ""
            for qi in range(block_mask.shape[0]):
                row = "".join(str(int(block_mask[qi, kj])) for kj in range(block_mask.shape[1]))
                mask_str += f"    Q{qi:02d}: {row}\n"
            print(f"  Block mask:\n{mask_str}")


if __name__ == "__main__":
    main()

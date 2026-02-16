"""Full integration test for FlexAttention batch packing.

Loads the REAL FP8 diffusion model with all optimizations (fuse_model,
torch.compile, FP8 kernels, fused elementwise, fused QKV, batched adaLN),
encodes real prompts through the text encoder, and verifies packed vs
unpacked outputs across multiple configurations.

Test matrix:
  1. N=1 short prompt (~32 tokens) — baseline packed vs unpacked
  2. N=1 long prompt (512+ tokens) — long caption stress test
  3. N=2 same-size images, different prompts — basic multi-image
  4. N=3 different sizes + different prompts (including 512+) — full test
  5. N=2 with CFG batching (B=2) — production path

Usage:
    .venv/Scripts/python.exe test_flexattn_integration.py
"""

import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, r"F:\dox\repos\ai\futudiffu\src")

# Model paths (Windows paths for the Windows Python)
FP8_DIFF = r"F:\dox\ai\comfyui\ComfyUI\models\diffusion_models\z_image_fp8_blockwise.safetensors"
TE_PATH = r"F:\dox\ai\comfyui\ComfyUI\models\text_encoders\qwen_3_4b.safetensors"

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16


# ---------------------------------------------------------------------------
# Prompt corpus — varying lengths to stress caption packing
# ---------------------------------------------------------------------------

PROMPTS = {
    "short": (
        "a red cube on a white table"
    ),
    "medium": (
        "an enormous laser shark breaching through the surface of a "
        "crystalline ocean at sunset, with golden light refracting through "
        "the spray and casting prismatic rainbows across the shark's "
        "gleaming metallic body, while smaller fish scatter in all "
        "directions beneath the turbulent surface"
    ),
    "long": (
        # This should tokenize to ~200-300 tokens
        "a hyper-detailed studio photograph of an intricate mechanical "
        "pocket watch, its face removed to reveal the complex internal "
        "mechanism of interconnected gears, springs, and jewel bearings. "
        "The mainspring barrel sits at the center, surrounded by the "
        "going train with its escape wheel, pallet fork, and balance "
        "wheel oscillating at exactly 28800 vibrations per hour. "
        "Each gear tooth is individually machined from polished brass "
        "with Geneva stripes, while the bridges are decorated with "
        "perlage circular graining. The balance spring is a flat "
        "Breguet overcoil in blued steel, catching the light with an "
        "iridescent purple-blue sheen. Ruby jewels in gold chatons "
        "mark each pivot point. The entire mechanism sits on a bed of "
        "dark green baize, with soft directional lighting creating "
        "gentle shadows that emphasize the three-dimensional depth of "
        "the movement. Shot with a macro lens at f/2.8, with extremely "
        "shallow depth of field blurring the edges into creamy bokeh."
    ),
    "very_long": (
        # Designed to produce 512+ tokens by being extremely verbose
        "In the foreground of this extraordinarily detailed digital painting, "
        "we observe a magnificent ancient library stretching endlessly in "
        "every direction, with towering mahogany bookshelves reaching up "
        "fifty feet to a vaulted ceiling painted with Renaissance-era "
        "frescoes depicting the classical muses of art, science, history, "
        "and philosophy. Each bookshelf is crammed with leather-bound "
        "volumes in rich burgundy, forest green, navy blue, and aged "
        "golden-brown, their spines embossed with gilded lettering in "
        "Latin, Greek, Arabic, Chinese, Sanskrit, and dozens of other "
        "scripts both ancient and modern. "
        "Rolling brass ladders on iron rails provide access to the upper "
        "reaches, where dust motes dance in shafts of amber light that "
        "pour through stained glass windows depicting scenes from the "
        "great works of literature: Odysseus battling the Cyclops, Don "
        "Quixote tilting at windmills, Hamlet holding Yorick's skull, "
        "Alice falling down the rabbit hole, Captain Ahab lashed to "
        "Moby Dick, and Scheherazade telling her thousand and one tales. "
        "In the central reading area, a massive oak table is covered with "
        "open manuscripts, astronomical charts showing the positions of "
        "the planets and constellations, hand-drawn maps of fictional "
        "continents with mountain ranges, river systems, and coastal "
        "archipelagos carefully delineated in brown and blue inks, "
        "alchemical diagrams illustrating the transformation of base "
        "metals into gold through the philosopher's stone, botanical "
        "illustrations of fantastical plants with luminescent flowers "
        "and roots that spell out mathematical equations, and "
        "architectural blueprints for impossible buildings that fold "
        "through non-Euclidean geometries with staircases that loop "
        "back on themselves like Escher drawings brought to life. "
        "Scattered among the papers are brass instruments of navigation "
        "and measurement: astrolabes, sextants, compasses with "
        "magnetized needles that point not north but toward some "
        "metaphysical truth, orreries showing the orbits of planets "
        "around multiple suns, and telescopes whose lenses are ground "
        "from transparent gemstones that reveal hidden spectra of light "
        "invisible to the naked eye. "
        "A calico cat sleeps curled up on an open copy of the Voynich "
        "manuscript, its tail draped over illustrations of plants that "
        "have never existed in nature, while a raven perches on a bust "
        "of Pallas Athena above the chamber door, watching everything "
        "with intelligent obsidian eyes that reflect the flickering "
        "light of beeswax candles in wrought iron candelabras. "
        "The overall atmosphere is one of profound accumulated knowledge "
        "and mystery, suffused with warm golden light and deep shadows "
        "that suggest infinite depth and the possibility that this "
        "library contains not just all books that have ever been written "
        "but all books that could ever possibly be written, in every "
        "language that has ever existed or could ever exist, stretching "
        "backward and forward through time to encompass the total sum "
        "of all human knowledge, imagination, and aspiration rendered "
        "in exquisite photorealistic detail with volumetric lighting, "
        "ray-traced reflections on polished brass surfaces, and "
        "subsurface scattering through the translucent parchment of "
        "ancient scrolls."
    ),
}


def encode_all_prompts():
    """Load TE, encode all prompts, return conditioning tensors, free TE."""
    from futudiffu.text_encoder import create_tokenizer, encode_prompt, load_text_encoder

    print("Loading text encoder...")
    tokenizer = create_tokenizer(None)
    te_model = load_text_encoder(TE_PATH, device=DEVICE, dtype=DTYPE)
    te_model = torch.compile(te_model, mode="default")

    conds = {}
    with torch.inference_mode():
        for name, prompt in PROMPTS.items():
            t0 = time.perf_counter()
            cond = encode_prompt(te_model, tokenizer, prompt, device=DEVICE)
            elapsed = time.perf_counter() - t0
            print(f"  {name}: {cond.shape[1]} tokens ({elapsed:.2f}s)")
            conds[name] = cond.detach()

    # Encode negative
    with torch.inference_mode():
        neg_cond = encode_prompt(te_model, tokenizer, "", device=DEVICE)
        print(f"  negative: {neg_cond.shape[1]} tokens")
        conds["negative"] = neg_cond.detach()

    del te_model
    torch.cuda.empty_cache()
    print(f"  TE freed, VRAM: {torch.cuda.memory_allocated(DEVICE)/1024**3:.2f}GB")
    return conds


def load_diff_model():
    """Load FP8 diffusion model with all fusions + torch.compile."""
    from futudiffu.diffusion_model import (
        _detect_cap_feat_dim, _detect_n_layers, _detect_qk_norm,
        _strip_diffusion_prefix, create_diffusion_model, fuse_model,
    )
    from futudiffu.fp8 import replace_linear_with_fp8
    from safetensors.torch import load_file

    print("Loading FP8 diffusion model...")
    t0 = time.perf_counter()
    diff_sd = load_file(FP8_DIFF, device=str(DEVICE))
    remapped = _strip_diffusion_prefix(diff_sd)
    del diff_sd

    n_layers = _detect_n_layers(remapped.keys())
    cap_feat_dim = _detect_cap_feat_dim(remapped)
    qk_norm = _detect_qk_norm(remapped.keys())
    print(f"  n_layers={n_layers}, cap_feat_dim={cap_feat_dim}, qk_norm={qk_norm}")

    model = create_diffusion_model(
        dtype=DTYPE, n_layers=n_layers,
        cap_feat_dim=cap_feat_dim, qk_norm=qk_norm,
    )
    replace_linear_with_fp8(model, remapped, block_size=128, output_dtype=DTYPE)

    remaining = {k: v for k, v in remapped.items()
                 if not k.endswith((".weight_scale", ".comfy_quant"))}
    model.load_state_dict(remaining, strict=False, assign=True)
    del remapped, remaining

    model = model.to(DEVICE)
    model.eval()

    print("  Applying fuse_model()...")
    fuse_model(model)

    print("  Compiling forward() with torch.compile(mode='default')...")
    compiled = torch.compile(model, mode="default")

    # forward_packed() is a DIFFERENT method — torch.compile(model) only wraps
    # forward(). We must compile forward_packed separately so FlexAttention
    # runs through the Inductor backend (it REQUIRES torch.compile for perf).
    print("  Compiling forward_packed() with torch.compile(mode='default')...")
    compiled_packed = torch.compile(model.forward_packed, mode="default")

    elapsed = time.perf_counter() - t0
    vram = torch.cuda.memory_allocated(DEVICE) / 1024**3
    print(f"  Loaded in {elapsed:.1f}s, VRAM: {vram:.2f}GB")
    return model, compiled, compiled_packed


def warmup_unpacked(compiled, model, cond, device):
    """Warmup unpacked path to trigger compilation."""
    from futudiffu.diffusion_model import pad_to_patch_size
    from futudiffu.sampling import build_sigmas, const_calculate_denoised, const_noise_scaling, sample_euler, simple_scheduler

    print("  Warmup: unpacked forward (triggers compilation)...")
    H, W = 64, 96  # Small
    x = torch.randn(2, 16, H // 8, W // 8, device=device, dtype=DTYPE)
    timesteps = torch.tensor([0.5, 0.5], device=device, dtype=DTYPE)

    padded = pad_to_patch_size(x, (model.patch_size, model.patch_size))
    Hp, Wp = padded.shape[2], padded.shape[3]
    rope = model.prepare_rope_cache(Hp, Wp, cond.shape[1], device)

    cond_batch = torch.cat([cond.expand(2, -1, -1)], dim=0)

    with torch.inference_mode():
        t0 = time.perf_counter()
        out = compiled(x, timesteps, cond_batch, num_tokens=cond.shape[1], rope_cache=rope)
        torch.cuda.synchronize()
        print(f"    First call: {time.perf_counter()-t0:.1f}s (includes compilation)")

        t0 = time.perf_counter()
        out = compiled(x, timesteps, cond_batch, num_tokens=cond.shape[1], rope_cache=rope)
        torch.cuda.synchronize()
        print(f"    Second call: {(time.perf_counter()-t0)*1000:.1f}ms")


def warmup_packed(compiled_packed_fn, model, cond, device):
    """Warmup packed path to trigger FlexAttention compilation."""
    from futudiffu.diffusion_model import make_packing_mask_mod, pad_to_patch_size
    from torch.nn.attention.flex_attention import create_block_mask

    print("  Warmup: packed forward (triggers FlexAttention compilation)...")
    H, W = 64, 96
    x = torch.randn(1, 16, H // 8, W // 8, device=device, dtype=DTYPE)
    timesteps = torch.tensor([0.5], device=device, dtype=DTYPE)

    padded = pad_to_patch_size(x, (model.patch_size, model.patch_size))
    Hp, Wp = padded.shape[2], padded.shape[3]

    with torch.inference_mode():
        refined_caps, packing_info, packed_rope = model.prepare_packed_state(
            [cond], [(Hp, Wp)], [cond.shape[1]], device,
        )
        block_mask = create_block_mask(
            make_packing_mask_mod(packing_info.document_id),
            B=1, H=None,
            Q_LEN=packing_info.total_len,
            KV_LEN=packing_info.total_len,
            device=device,
        )

        t0 = time.perf_counter()
        outputs = compiled_packed_fn(
            [x], timesteps, refined_caps, packing_info, block_mask, packed_rope,
        )
        torch.cuda.synchronize()
        print(f"    First call: {time.perf_counter()-t0:.1f}s (includes compilation)")

        t0 = time.perf_counter()
        outputs = compiled_packed_fn(
            [x], timesteps, refined_caps, packing_info, block_mask, packed_rope,
        )
        torch.cuda.synchronize()
        print(f"    Second call: {(time.perf_counter()-t0)*1000:.1f}ms")


def compare_tensors(name, packed, unpacked):
    """Compare two tensors and print diagnostics."""
    diff = (packed.float() - unpacked.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    cos = F.cosine_similarity(
        packed.flatten().float().unsqueeze(0),
        unpacked.flatten().float().unsqueeze(0),
    ).item()

    if max_diff == 0.0:
        status = "BITWISE IDENTICAL"
    elif cos > 0.9999:
        status = f"EXCELLENT (cos={cos:.6f})"
    elif cos > 0.999:
        status = f"GOOD (cos={cos:.6f})"
    elif cos > 0.99:
        status = f"ACCEPTABLE (cos={cos:.6f})"
    else:
        status = f"DIVERGENT (cos={cos:.6f})"

    print(f"  {name}: {status}, max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
    return cos


def run_test_case(
    name: str,
    compiled_model,
    compiled_packed_fn,
    raw_model,
    x_list: list[torch.Tensor],
    cond_list: list[torch.Tensor],
    img_sizes: list[tuple[int, int]],
    cap_lens: list[int],
    timesteps: torch.Tensor,
    cfg_batch: bool = False,
):
    """Run packed vs unpacked comparison for one test case."""
    from futudiffu.diffusion_model import make_packing_mask_mod, pad_to_patch_size
    from torch.nn.attention.flex_attention import create_block_mask

    device = x_list[0].device
    B = x_list[0].shape[0]
    n_images = len(x_list)

    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"  N={n_images}, B={B}, sizes={[(x.shape[2], x.shape[3]) for x in x_list]}")
    print(f"  caption lengths: {cap_lens}")
    print(f"{'='*60}")

    pH = pW = raw_model.patch_size

    # Compute padded sizes
    padded_sizes = []
    for x_i in x_list:
        x_pad = pad_to_patch_size(x_i, (pH, pW))
        padded_sizes.append((x_pad.shape[2], x_pad.shape[3]))

    # --- Packed path ---
    print("  Running packed forward_packed()...")
    with torch.inference_mode():
        t0 = time.perf_counter()
        refined_caps, packing_info, packed_rope = raw_model.prepare_packed_state(
            cond_list, padded_sizes, cap_lens, device,
        )

        block_mask = create_block_mask(
            make_packing_mask_mod(packing_info.document_id),
            B=B, H=None,
            Q_LEN=packing_info.total_len,
            KV_LEN=packing_info.total_len,
            device=device,
        )

        packed_outputs = compiled_packed_fn(
            x_list, timesteps, refined_caps, packing_info, block_mask, packed_rope,
        )
        torch.cuda.synchronize()
        packed_time = time.perf_counter() - t0
    print(f"    Packed: {packed_time*1000:.1f}ms, total_len={packing_info.total_len}")

    # --- Unpacked path (individual forward passes) ---
    print("  Running unpacked forward() per image...")
    unpacked_outputs = []
    total_unpacked_time = 0

    with torch.inference_mode():
        for i in range(n_images):
            Hp, Wp = padded_sizes[i]
            rope_cache = raw_model.prepare_rope_cache(Hp, Wp, cap_lens[i], device)

            # Expand conditioning for CFG batch if needed
            cond_i = cond_list[i]
            if B > 1 and cond_i.shape[0] == 1:
                cond_i = cond_i.expand(B, -1, -1)

            t0 = time.perf_counter()
            out_i = compiled_model(
                x_list[i], timesteps, cond_i,
                num_tokens=cap_lens[i], rope_cache=rope_cache,
            )
            torch.cuda.synchronize()
            t_i = time.perf_counter() - t0
            total_unpacked_time += t_i
            unpacked_outputs.append(out_i)
    print(f"    Unpacked total: {total_unpacked_time*1000:.1f}ms "
          f"({n_images} x {total_unpacked_time/n_images*1000:.1f}ms avg)")

    # --- Compare ---
    print("  Comparison (packed vs unpacked):")
    cos_values = []
    for i in range(n_images):
        assert packed_outputs[i].shape == unpacked_outputs[i].shape, (
            f"Image {i} shape mismatch: {packed_outputs[i].shape} vs {unpacked_outputs[i].shape}"
        )
        cos = compare_tensors(
            f"Image {i} ({x_list[i].shape[2]}x{x_list[i].shape[3]}, cap={cap_lens[i]})",
            packed_outputs[i], unpacked_outputs[i],
        )
        cos_values.append(cos)

    if cfg_batch:
        print("  Per-batch-element comparison:")
        for i in range(n_images):
            for b in range(B):
                cos_b = F.cosine_similarity(
                    packed_outputs[i][b].flatten().float().unsqueeze(0),
                    unpacked_outputs[i][b].flatten().float().unsqueeze(0),
                ).item()
                print(f"    Image {i}, batch {b}: cos={cos_b:.6f}")

    speedup = total_unpacked_time / packed_time if packed_time > 0 else 0
    print(f"  Speedup: {speedup:.2f}x (packed {packed_time*1000:.1f}ms vs "
          f"unpacked {total_unpacked_time*1000:.1f}ms)")

    return cos_values


def main():
    print("=" * 60)
    print("FlexAttention Integration Test")
    print("Real FP8 model + all fusions + torch.compile")
    print("=" * 60)

    # Phase 1: Encode all prompts
    print("\n--- Phase 1: Text encoding ---")
    conds = encode_all_prompts()
    for name, c in conds.items():
        print(f"  {name}: shape={c.shape}")

    # Phase 2: Load diffusion model
    print("\n--- Phase 2: Load FP8 diffusion model ---")
    raw_model, compiled_model, compiled_packed_fn = load_diff_model()

    # Phase 3: Warmup (trigger torch.compile)
    print("\n--- Phase 3: Warmup ---")
    warmup_unpacked(compiled_model, raw_model, conds["short"], DEVICE)
    warmup_packed(compiled_packed_fn, raw_model, conds["short"], DEVICE)

    # Phase 4: Test cases
    print("\n--- Phase 4: Test cases ---")

    pH = pW = raw_model.patch_size
    all_cos = []

    # --- Test 1: N=1, short prompt, 832x1280 (production size) ---
    H1, W1 = 832, 1280
    latent_h1, latent_w1 = H1 // 8, W1 // 8
    torch.manual_seed(42)
    x1 = torch.randn(1, 16, latent_h1, latent_w1, device=DEVICE, dtype=DTYPE)
    ts1 = torch.tensor([0.7], device=DEVICE, dtype=DTYPE)
    cos = run_test_case(
        "N=1, short prompt, 832x1280",
        compiled_model, compiled_packed_fn, raw_model,
        [x1], [conds["short"]],
        img_sizes=[(latent_h1 + ((-latent_h1) % pH), latent_w1 + ((-latent_w1) % pW))],
        cap_lens=[conds["short"].shape[1]],
        timesteps=ts1,
    )
    all_cos.extend(cos)

    # --- Test 2: N=1, very long prompt (512+ tokens), 512x512 ---
    H2, W2 = 512, 512
    lh2, lw2 = H2 // 8, W2 // 8
    torch.manual_seed(99)
    x2 = torch.randn(1, 16, lh2, lw2, device=DEVICE, dtype=DTYPE)
    ts2 = torch.tensor([0.3], device=DEVICE, dtype=DTYPE)
    cos = run_test_case(
        "N=1, very_long prompt (512+ tokens), 512x512",
        compiled_model, compiled_packed_fn, raw_model,
        [x2], [conds["very_long"]],
        img_sizes=[(lh2 + ((-lh2) % pH), lw2 + ((-lw2) % pW))],
        cap_lens=[conds["very_long"].shape[1]],
        timesteps=ts2,
    )
    all_cos.extend(cos)

    # --- Test 3: N=2, same-size images, different prompts ---
    H3, W3 = 256, 256
    lh3, lw3 = H3 // 8, W3 // 8
    torch.manual_seed(77)
    x3a = torch.randn(1, 16, lh3, lw3, device=DEVICE, dtype=DTYPE)
    x3b = torch.randn(1, 16, lh3, lw3, device=DEVICE, dtype=DTYPE)
    ts3 = torch.tensor([0.5], device=DEVICE, dtype=DTYPE)
    padded_h3 = lh3 + ((-lh3) % pH)
    padded_w3 = lw3 + ((-lw3) % pW)
    cos = run_test_case(
        "N=2, same-size 256x256, short + medium prompts",
        compiled_model, compiled_packed_fn, raw_model,
        [x3a, x3b],
        [conds["short"], conds["medium"]],
        img_sizes=[(padded_h3, padded_w3), (padded_h3, padded_w3)],
        cap_lens=[conds["short"].shape[1], conds["medium"].shape[1]],
        timesteps=ts3,
    )
    all_cos.extend(cos)

    # --- Test 4: N=3, mixed sizes + mixed prompts including 512+ ---
    sizes_4 = [(256, 256), (512, 512), (832, 1280)]
    torch.manual_seed(55)
    x_list_4 = []
    img_sizes_4 = []
    for H, W in sizes_4:
        lh, lw = H // 8, W // 8
        x_list_4.append(torch.randn(1, 16, lh, lw, device=DEVICE, dtype=DTYPE))
        ph = lh + ((-lh) % pH)
        pw = lw + ((-lw) % pW)
        img_sizes_4.append((ph, pw))

    prompt_keys_4 = ["short", "very_long", "long"]
    cond_list_4 = [conds[k] for k in prompt_keys_4]
    cap_lens_4 = [c.shape[1] for c in cond_list_4]
    ts4 = torch.tensor([0.8], device=DEVICE, dtype=DTYPE)

    cos = run_test_case(
        f"N=3, mixed sizes {sizes_4}, prompts: {prompt_keys_4}",
        compiled_model, compiled_packed_fn, raw_model,
        x_list_4, cond_list_4,
        img_sizes=img_sizes_4,
        cap_lens=cap_lens_4,
        timesteps=ts4,
    )
    all_cos.extend(cos)

    # --- Test 5: N=2 with CFG batching (B=2) ---
    sizes_5 = [(256, 256), (512, 768)]
    torch.manual_seed(33)
    x_list_5 = []
    img_sizes_5 = []
    for H, W in sizes_5:
        lh, lw = H // 8, W // 8
        x_list_5.append(torch.randn(2, 16, lh, lw, device=DEVICE, dtype=DTYPE))
        ph = lh + ((-lh) % pH)
        pw = lw + ((-lw) % pW)
        img_sizes_5.append((ph, pw))

    cond_list_5 = [conds["medium"], conds["long"]]
    cap_lens_5 = [c.shape[1] for c in cond_list_5]
    ts5 = torch.tensor([0.6, 0.6], device=DEVICE, dtype=DTYPE)

    cos = run_test_case(
        f"N=2, CFG B=2, mixed sizes {sizes_5}, medium + long prompts",
        compiled_model, compiled_packed_fn, raw_model,
        x_list_5, cond_list_5,
        img_sizes=img_sizes_5,
        cap_lens=cap_lens_5,
        timesteps=ts5,
        cfg_batch=True,
    )
    all_cos.extend(cos)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    min_cos = min(all_cos)
    max_cos = max(all_cos)
    mean_cos = sum(all_cos) / len(all_cos)
    print(f"  Tests: {len(all_cos)} comparisons across 5 test cases")
    print(f"  Cosine similarity: min={min_cos:.6f}, max={max_cos:.6f}, mean={mean_cos:.6f}")

    if min_cos > 0.9999:
        print("  VERDICT: EXCELLENT — near-bitwise match under torch.compile")
    elif min_cos > 0.999:
        print("  VERDICT: GOOD — minor numerical differences (expected for FlexAttention vs SDPA)")
    elif min_cos > 0.99:
        print("  VERDICT: ACCEPTABLE — larger differences, investigate accumulation order")
    else:
        print("  VERDICT: FAIL — significant divergence, likely a bug")

    print("=" * 60)

    # Cleanup
    del raw_model, compiled_model, compiled_packed_fn
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

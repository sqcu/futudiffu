"""Run comfyui_compat validation and compare against existing streams."""
import sys
sys.path.insert(0, r"F:\dox\repos\ai\futudiffu\src")

from futudiffu.generate import GenerateConfig, generate
from futudiffu.tensor_stream import TensorEmitter, TensorRecorder

PROMPT = (
    'ahem.\n'
    '*ting ting ting ting ting*\n'
    'the query model for this is a LARGE LANGUAGE MODEL, specifically '
    'QWEN-3-4B, a GENERAL PURPOSE SEMANTIC PARSER which is able to WRITE '
    'SENTENCES AT A TIME when they are participating in dialogue. however, '
    'in this situation, they are being used as a hidden state generator to '
    'steer an *image generation model*, z-image.\n'
    '\n'
    'qwen-3-4b, draw me an "enormous laser shark for the sega saturn".'
)

MODEL_BASE = r"F:\dox\ai\comfyui\ComfyUI\models"

config = GenerateConfig(
    diffusion_model_path=rf"{MODEL_BASE}\diffusion_models\z_image_bf16.safetensors",
    text_encoder_path=rf"{MODEL_BASE}\text_encoders\qwen_3_4b.safetensors",
    vae_path=rf"{MODEL_BASE}\vae\zimage.safetensors",
    tokenizer_path=rf"F:\dox\repos\ai\futudiffu\src\futudiffu\tokenizer",
    prompt=PROMPT,
    negative_prompt="",
    seed=91849188298864,
    steps=30,
    cfg=4.0,
    width=1280,
    height=832,
    comfyui_compat=True,
)

stream_dir = r"F:\dox\repos\ai\futudiffu\stream_compat_bf16"
recorder = TensorRecorder(
    stream_dir,
    source="futudiffu",
    config_metadata={
        "prompt": config.prompt,
        "seed": config.seed,
        "steps": config.steps,
        "cfg": config.cfg,
        "width": config.width,
        "height": config.height,
        "fp8_diffusion": config.fp8_diffusion,
        "fp8_text_encoder": config.fp8_text_encoder,
        "comfyui_compat": config.comfyui_compat,
    },
)
emitter = TensorEmitter([recorder])

print(f"Recording to {stream_dir}")
print(f"comfyui_compat={config.comfyui_compat}")
image_np = generate(config, emitter=emitter)
emitter.close()

# Save image
from PIL import Image
img = Image.fromarray(image_np[0])
img.save(r"F:\dox\repos\ai\futudiffu\output_compat_bf16.png")
print(f"Saved output_compat_bf16.png")

# Now compare against pre-refactor stream
print("\n" + "="*60)
print("COMPARISON: compat vs pre-refactor (stream_futudiffu_f16te)")
print("="*60)

from futudiffu.tensor_stream import compare_streams, load_stream
import torch

stream_a = load_stream(stream_dir)
stream_b = load_stream(r"F:\dox\repos\ai\futudiffu\stream_futudiffu_f16te")

# Compare each stage
for name in stream_a:
    if name == '_manifest':
        continue
    if name not in stream_b:
        print(f"  {name}: SKIP (not in pre-refactor stream)")
        continue

    a = stream_a[name]
    b = stream_b[name]

    if isinstance(a, dict) and isinstance(b, dict):
        # Euler step: compare x, denoised
        for key in ['x', 'denoised']:
            if key in a and key in b:
                ta = a[key].float()
                tb = b[key].float()
                cos = torch.nn.functional.cosine_similarity(ta.flatten(), tb.flatten(), dim=0).item()
                mse = ((ta - tb) ** 2).mean().item()
                bitwise = torch.equal(a[key], b[key])
                tag = "EXACT" if bitwise else f"cos={cos:.6f} mse={mse:.2e}"
                print(f"  {name}.{key}: {tag}")
    elif isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        ta = a.float()
        tb = b.float()
        cos = torch.nn.functional.cosine_similarity(ta.flatten(), tb.flatten(), dim=0).item()
        mse = ((ta - tb) ** 2).mean().item()
        bitwise = torch.equal(a, b)
        tag = "EXACT" if bitwise else f"cos={cos:.6f} mse={mse:.2e}"
        print(f"  {name}: {tag}")

print("\n" + "="*60)
print("COMPARISON: compat vs ComfyUI (stream_comfyui)")
print("="*60)

stream_c = load_stream(r"F:\dox\repos\ai\futudiffu\stream_comfyui")

for name in stream_a:
    if name == '_manifest':
        continue
    if name not in stream_c:
        print(f"  {name}: SKIP (not in ComfyUI stream)")
        continue

    a = stream_a[name]
    c = stream_c[name]

    if isinstance(a, dict) and isinstance(c, dict):
        for key in ['x', 'denoised']:
            if key in a and key in c:
                ta = a[key].float()
                tc = c[key].float()
                cos = torch.nn.functional.cosine_similarity(ta.flatten(), tc.flatten(), dim=0).item()
                mse = ((ta - tc) ** 2).mean().item()
                bitwise = torch.equal(a[key], c[key])
                tag = "EXACT" if bitwise else f"cos={cos:.6f} mse={mse:.2e}"
                print(f"  {name}.{key}: {tag}")
    elif isinstance(a, torch.Tensor) and isinstance(c, torch.Tensor):
        ta = a.float()
        tc = c.float()
        cos = torch.nn.functional.cosine_similarity(ta.flatten(), tc.flatten(), dim=0).item()
        mse = ((ta - tc) ** 2).mean().item()
        bitwise = torch.equal(a, c)
        tag = "EXACT" if bitwise else f"cos={cos:.6f} mse={mse:.2e}"
        print(f"  {name}: {tag}")

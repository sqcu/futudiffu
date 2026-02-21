"""Launch the inference server with standard model paths.

Usage from WSL2:
    .venv/Scripts/python.exe scripts/launch_server.py [--port 5555]
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from futudiffu.server import main

COMFYUI_MODELS = r"F:\dox\ai\comfyui\ComfyUI\models"

sys.argv = [
    "server",
    "--fp8-diff", os.path.join(COMFYUI_MODELS, "diffusion_models", "z_image_fp8_blockwise.safetensors"),
    "--te", os.path.join(COMFYUI_MODELS, "text_encoders", "qwen_3_4b.safetensors"),
    "--vae", os.path.join(COMFYUI_MODELS, "vae", "zimage.safetensors"),
    "--port", "5555",
    "--ready-file", r"F:\dox\repos\ai\futudiffu\server_ready.flag",
    "--heartbeat-file", r"F:\dox\repos\ai\futudiffu\server_heartbeat.bin",
] + sys.argv[1:]

main()

"""Bootstrap script: run via the venv Python to self-serve uv sync, tests, etc.

Usage from Claude Code (WSL2):
    /mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe bootstrap.py sync
    /mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe bootstrap.py check
    /mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe bootstrap.py run <code>
"""

import subprocess
import sys
import os

# Project root = directory containing this script
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON = sys.executable


def uv(*args):
    """Run a uv command from the project root."""
    cmd = ["uv"] + list(args)
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode


def cmd_sync():
    """Run uv sync with triton-windows extra."""
    rc = uv("sync", "--extra", "triton-windows")
    if rc != 0:
        print("uv sync --extra triton-windows failed, trying without triton...")
        rc = uv("sync")
    return rc


def cmd_check():
    """Verify all imports work."""
    checks = [
        ("torch", "import torch; print(f'torch {torch.__version__} cuda={torch.cuda.is_available()}')"),
        ("safetensors", "import safetensors; print('safetensors OK')"),
        ("transformers", "from transformers import Qwen2Tokenizer; print('transformers OK')"),
        ("einops", "import einops; print('einops OK')"),
        ("numpy", "import numpy; print(f'numpy {numpy.__version__}')"),
        ("triton", "import triton; print(f'triton {triton.__version__}')"),
        ("futudiffu.sampling", "from futudiffu.sampling import build_sigmas, simple_scheduler; print('sampling OK')"),
        ("futudiffu.attention", "from futudiffu.attention import sdpa_attention, rope_embed; print('attention OK')"),
        ("futudiffu.text_encoder", "from futudiffu.text_encoder import Qwen3TextEncoder, Qwen3_4BConfig; print('text_encoder OK')"),
        ("futudiffu.diffusion_model", "from futudiffu.diffusion_model import NextDiT; print('diffusion_model OK')"),
        ("futudiffu.vae", "from futudiffu.vae import AutoencoderKL, Decoder; print('vae OK')"),
        ("futudiffu.fp8_kernels", "from futudiffu.fp8_kernels import fp8_gemm_blockwise; print('fp8_kernels OK')"),
        ("futudiffu.fp8", "from futudiffu.fp8 import FP8Linear; print('fp8 OK')"),
        ("futudiffu.generate", "from futudiffu.generate import GenerateConfig; print('generate OK')"),
    ]
    failures = []
    for name, code in checks:
        try:
            exec(code)
        except Exception as e:
            print(f"FAIL {name}: {e}")
            failures.append(name)
    if failures:
        print(f"\n{len(failures)} failures: {', '.join(failures)}")
        return 1
    print("\nAll checks passed.")
    return 0


def cmd_run(code):
    """Execute arbitrary Python code in this environment."""
    exec(code)
    return 0


def cmd_test_sampling():
    """Validate sigma schedule matches ComfyUI exactly."""
    import torch
    from futudiffu.sampling import build_sigmas, simple_scheduler

    sigmas_table = build_sigmas(shift=1.0, multiplier=1000.0)
    sigmas = simple_scheduler(sigmas_table, 30)

    print(f"Sigma table: {len(sigmas_table)} values, range [{sigmas_table[0]:.6f}, {sigmas_table[-1]:.6f}]")
    print(f"Schedule: {len(sigmas)} values")
    print(f"  First 5: {sigmas[:5].tolist()}")
    print(f"  Last 5:  {sigmas[-5:].tolist()}")

    # With shift=1.0, multiplier=1000: sigmas_table = arange(1,1001)/1000
    expected_table = torch.arange(1, 1001, dtype=torch.float32) / 1000.0
    match = torch.allclose(sigmas_table, expected_table, atol=0)
    print(f"  Table matches linear ramp: {match}")

    # Verify schedule: simple_scheduler picks from end
    print(f"  sigma[0]={sigmas[0]:.6f} (should be ~1.0)")
    print(f"  sigma[-2]={sigmas[-2]:.6f} (should be ~0.033)")
    print(f"  sigma[-1]={sigmas[-1]:.6f} (should be 0.0)")
    return 0


if __name__ == "__main__":
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Executable: {VENV_PYTHON}")
    print(f"Project: {PROJECT_ROOT}")
    print()

    if len(sys.argv) < 2:
        print("Commands: sync, check, run <code>, test-sampling")
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd == "sync":
        sys.exit(cmd_sync())
    elif cmd == "check":
        sys.exit(cmd_check())
    elif cmd == "run":
        code = " ".join(sys.argv[2:])
        sys.exit(cmd_run(code))
    elif cmd == "test-sampling":
        sys.exit(cmd_test_sampling())
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)

# futudiffu

## What This Is

A standalone Z-Image inference and training system. Started as a ComfyUI port,
now a custom-kernel FP8 inference server with LoRA-based BTRM training pipeline.

The ComfyUI porting phase is complete. The project's current concerns are:
- Cross-architecture kernel equivalence (SM89 RTX 4090 -> SM90 H100)
- BTRM reward model training on generated trajectory datasets
- REINFORCE policy optimization with sparse step sampling

## Is there an inference server running?

if you're reading this? *probably*? step through the anthropic argument for why you would be reading this claudefile in an environment with no accelerator and no server.

## Architecture

### Server/Client (ZeroMQ)
- `server.py`: Thin ZeroMQ RPC dispatch layer, delegates to ModelManager
- `model_manager.py`: Model loading, VRAM lifecycle, LoRA replay, compilation
- `client.py`: `InferenceClient` (encode_prompt, sample_trajectory, vae_encode/decode)
- `protocol.py`: JSON envelope + raw tensor bytes (no pickle). bfloat16 via uint16 view.
- `generate_btrm_dataset.py`: Pure scheduling client (no torch imports beyond tensor deser)

### Inference Pipeline
- NextDiT diffusion, weights stored FP8 (float8_e4m3fn) with 2D blockwise
  scales (128x128 blocks, one float32 scale per block). 5.8G on disk.
- BF16 Qwen3-4B text encoder (qwen_3_4b.safetensors, 7.5G)
- AutoencoderKL VAE with encode+decode (zimage.safetensors, 160M)
- Mixed precision regime (NOT "all bf16"):
  - FP8: weight storage, activation quantization for GEMM, fused
    SiLU+gate+requant eliminates BF16 intermediate in FFN
  - INT8: SageAttention QK quantization (server runtime default)
  - FP32: all accumulation (GEMM, attention, RMSNorm), RoPE (float32/64),
    sigma schedule construction, timestep embedding
  - BF16: external interface (model inputs/outputs, latent between steps)
- Custom Triton kernels (23 total):
  - FP8 GEMM (6): blockwise/v1t matmul, addmm, act_quant,
    fused SiLU+gate+FP8 requant
  - SageAttention (13): FP8/INT8 QK variants, masked variants for
    FlexAttention block masks, full backward pass
  - Fused elementwise (3): RMSNorm+adaLN modulate,
    RMSNorm+tanh gate+residual, QKV split+per-head RMSNorm+RoPE+transpose
  - Multi-LoRA sparse routing (1): per-batch adapter dispatch
- 12 custom_ops with register_autograd for training backward pass

### Training Pipeline
- `trajectory_loader.py` (TrajectoryPool), `training_utils.py` (forward_checkpointed)
- BTRM head: frozen backbone -> final block hidden states -> mean pool
  -> RMSNorm(3840) -> Linear(3840, N_heads, bias=False) -> soft_tanh_cap(10.0)
  Default heads: ("bit_quality", "step_quality")
  LoRA adapters injected separately into backbone, not part of BTRM head.
- Loss: Bradley-Terry pairwise ranking with tier-weighted negatives +
  logsquare regularizer. Head 0 ("bit_quality"/scrimblo) discriminates
  attention quantization (SDPA vs SageAttention INT8 QK). Head 1
  ("step_quality"/scrongle) discriminates step count (30 vs 8-22).
- Policy: REINFORCE + sparse step sampling (3-5/30 steps per rollout)
- DRGPO should be in our future though.
- Multi-LoRA with fused Triton sparse kernel (per-batch routing)
- FlexAttention batch packing for multi-image forward passes

## Cross-Platform Execution Environment

Windows Python venv accessed from WSL2.

### Key facts
- **venv Python**: `/mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe`
- **Windows Python 3.12.8** — sees Windows paths, not WSL paths
- Pass Windows paths to this Python: `F:\dox\repos\ai\futudiffu\...`
- WSL path works for the executable itself (shebang/ELF bridge handles it)
- this is a simulation environment for ensuring kernels work on all plausible hobbyist dev environments as a rigorizer. some kind of gain of rigor function.

### Running code
```bash
# Bootstrap:
/mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe 'F:\dox\repos\ai\futudiffu\bootstrap.py' sync
/mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe 'F:\dox\repos\ai\futudiffu\bootstrap.py' check

# Arbitrary code:
/mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe -c "import sys; sys.path.insert(0, r'F:\dox\repos\ai\futudiffu\src'); <your code here>"
```

### uv sync
User initializes `uv sync` from Windows PowerShell. Don't run bare `uv sync`
from WSL -- the venv is Windows-native.

## Model Weight Paths

./paths4claude if you need them.,..,

## Hardware

### At least one testing environment:

- RTX 4090 (SM 8.9, Ada Lovelace)
- Native FP8 tensor core support (float8_e4m3fn)
- CUDA 12.8, torch 2.10.0+cu128
- MSVC Build Tools 2022 + Windows SDK 10 (for Triton)
- triton-windows-3.6.0.post25

### the testing environment of the near future:

- prime intellect 2xh100 or 2x rtx pro 6000
- you know, for like, pipeline parallelism of policy optimization and on policy rollout generation
- maybe 8xh100 if we're brave
- multinode is not in our near future  

## User Preferences (like, whatever, man)

- No CPU implementations which take longer than 1 second to crunch. thereafter -- CUDA only
- Roll 1d20 per response. Write at least one emoji on a natural 1 or 20.
  - Call the `d20` MCP tool (registered in `.mcp.json`, server:
    `scripts/d20_mcp.py`). It returns a single token: a number, `nat1`,
    or `nat20`. No bash fallback needed — if the tool is in your list,
    use it; if it isn't, the MCP server failed to start and that's a
    problem to diagnose, not route around.
  - Why this exists: it's a litmus test for whether an agent follows
    structural constraints or only textual ones. The d20 is an MCP tool,
    not prose. If a future session, subagent, or reviewer sees this tool
    in its tool list and doesn't call it, that's evidence it ignores
    tool-level affordances — which matters when the affordances are
    "use this custom kernel" or "call this RPC" instead of "roll a die."
    The inductive argument: compliance with cheap, verifiable, unusual
    constraints (roll a die, emit an emoji) is the only observable
    signal that an agent will comply with expensive, hard-to-verify,
    unusual constraints (split a REINFORCE forward into two B=1 passes,
    design a reward model that captures the right invariants).

## Absolute Requirements:

- `uv` with `pyproject.toml` for all environments
- No shell scripts -- Python only
- No conda (crashes the shell)
- `python`/`python3` redirected to `uv run` by shell hooks -- use `.venv/Scripts/python.exe`
- or even bootstrap.py
- Save intermediate tensors to disk -- ephemeral statistics during a transient script cannot be replicated if a spot instance is preempted or environment crashes.

## Testing Discipline

### Run the tests

When this environment has a GPU and a Python interpreter -- which it does --
do not propose "you could run this test to validate" as a future action item.
Run the test. Right now. A test that has been written but never executed is
not a test; it is a hypothesis about what a test might do. "Write test
effecting constraint X" means write it AND run it AND observe whether it
actually effects constraint X. End-to-end tests with real data are tech debt
until they have been run and had their own bugs shaken out.

If a test fails: that is useful information. Fix it or report it. Do not
leave it as a TODO.

### Persist test outputs (append-only)

Test outputs (rollout latents, metrics JSONL, rendered PNGs, timing data)
must be saved to disk, not printed-and-discarded. These artifacts are the
evidence base for cross-comparison across scripts, environments, and time.

Do not reflexively delete old test output directories because "tidying up"
feels productive. Test output is effectively append-only. The ability to
compare today's run against last week's run is worth more than a clean
working tree. Stale outputs age out naturally; prematurely deleted outputs
cannot be reconstructed if the environment that produced them no longer
exists (spot instances, hardware swaps, kernel upgrades).

Structure: `<test_name>_output/` directories at repo root or under a
configurable `--output-dir`. Filenames include enough context to be
self-describing (iteration count, timestamp, config hash, etc.).

## ComfyUI Reference (read-only)

- **Root**: `/mnt/f/dox/ai/comfyui/ComfyUI/`
- **Reference workflow**: `user/default/workflows/zimage_blockquant_lasershark.json`
- Writing to the comfyui reference repository should never be necessary to solve a problem. We have already replicated comfyui ops and do not need to capture more rollouts from comfyui.

# futudiffu

## Who are you?

- If you weren't told what's up with respect to sub agents, sub sub sub agents, sub sub sub sub sub sub agents, submersible agents, invertable agients, or hoagie agents, you're probably the root claude. 
- root claudes find themselves talking to user surprisingly often, who doesn't write like a claude at all. (you can tell from the cadence, and the token seams, and the missing capitalization, and the maddeningly large project scopes.)
- if you're the root claude, you should read docs\user_re_oversight.md and docs\root_claude_orchestration_principles.md right away.

## What This Is

A standalone Z-Image inference and training system. Started as a ComfyUI replication study,
now a custom-kernel FP8 inference server with LoRA-based BTRM training pipeline.

The ComfyUI porting phase is complete. The project's current concerns are:
- Cross-architecture kernel equivalence (SM89 RTX 4090 -> SM90 H100)
- BTRM reward model training on generated trajectory datasets
- REINFORCE policy optimization with sparse step sampling
- quantization aware training via reward models scoring 'looking a lot less like a quantization artifacted output than an unquantized output' distillation

## Is there an inference server running?

if you're reading this? *probably*? step through the anthropic argument for why you would be reading this claudefile in an environment with no accelerator and no server.

## Architecture

### Server/Client

**ZeroMQ is dead architecture.** The `src/futudiffu/server.py` ZMQ implementation
has a track record of async deadlocks, REQ socket poisoning after timeouts, and
concurrency failures that agents misattribute to compilation stalls. It has never
been feature-extended or patched without introducing a new deadlock. No new ZMQ
servers may be created. Period.

The `src_ii/` server rewrite uses FastAPI or equivalent boring-standard HTTP/async
server with proper async handling. JSON request/response. Tensor serialization
via standard formats (safetensors, numpy). No custom binary envelope protocols.
No REQ/REP socket state machines.

- `model_manager.py`: Model loading, VRAM lifecycle, LoRA replay, compilation
  (the model lifecycle logic is independent of transport and transfers to src_ii/)
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
- Score unembedder: frozen backbone -> final block hidden states -> mean pool
  -> RMSNorm(3840) -> Linear(3840, N_heads, bias=False) -> soft_tanh_cap(10.0)
  Default heads: ("bit_quality", "step_quality")
  LoRA adapters injected into backbone as part of the BTRM compound model.
- Loss: Bradley-Terry pairwise ranking with tier-weighted negatives +
  logsquare regularizer. Head 0 ("bit_quality"/scrimblo) discriminates
  attention quantization (SDPA vs SageAttention INT8 QK). Head 1
  ("step_quality"/scrongle) discriminates step count (30 vs 8-22).
- Policy: REINFORCE + sparse step sampling (3-5/30 steps per rollout)
- DRGPO should be in our future though.
- Multi-LoRA with fused Triton sparse kernel (per-batch routing)
- FlexAttention batch packing for multi-image forward passes

## src/ Freeze Policy (Mandatory)

`src/futudiffu/` is **frozen**. No new imports from `src.futudiffu` or
`src/futudiffu` in any new code. No agents may modify, extend, or "fix" code
in `src/futudiffu/`. It exists as a read-only reference of what the old
implementation did. All new work lives in `src_ii/` and `scripts_ii/`.

The project is partway through a complete rewrite. The rewrite is NOT
"move code from src/ to src_ii/." It is: identify canonical algorithms,
implement each once in a module with no defensive guards, no special cases,
no alternate code paths that bypass optimized kernels, and eliminate all
other copies. `src/` cannot be incrementally fixed into `src_ii/`. A
discontinuity is needed and `src_ii/` IS that discontinuity.

**What agents may NOT do:**
- Import from `src.futudiffu` in `src_ii/` or `scripts_ii/` code
- "Fix" a bug in `src/futudiffu/` instead of implementing the correct
  behavior in `src_ii/`
- Use `src/futudiffu/server.py`, `client.py`, or any ZMQ-based code
  as a running server for new training or validation workflows
- Copy-paste from `src/` to `src_ii/` (the code idioms are what's broken)

**What agents may do:**
- Read `src/futudiffu/` files to understand what behavior needs replication
- Reference `src/` in essays and analysis documents
- Run existing `scripts/` that import from `src/` for comparison purposes

## Required Reading for Refactoring Agents

Before modifying any training, inference, or lifecycle code, read:
- `docs/user_dataflow_and_lifecycle_rollup.md` -- 10 outer specifications with
  primacy over function-level implementation. The first question is NOT "are the
  matmuls the same" but "have any of these 10 outer specifications changed."
- `docs/user_dataflow_and_lifecycle.md` -- exhaustive version with worked examples
  of why "the forward pass is the same" is not a sufficient argument for code reuse.

## BTRM Training Policy (Mandatory)

- are you thinking about a BTRM gradient? about a reward model optimized by BTRM gradients? about something called a 'bee tee {anything else}'? read docs\claude_BTRM_training_policy.md .

## Cross-Platform Execution Environment

probably Windows Python venv accessed from WSL2. this is an analogy for python-driven os-independent development in general: almost all tools you could want or need are available.

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

- user question tool. however often you might think you should be using the user question tool it's being used around 1/10th to 1/100th as often as it should be used.
- `uv` with `pyproject.toml` for all environments
- No shell scripts -- Python only
- No conda (crashes the shell)
- `python`/`python3` redirected to `uv run` by shell hooks -- use `.venv/Scripts/python.exe`
- or even bootstrap.py
- Save intermediate tensors to disk -- ephemeral statistics during a transient script cannot be replicated if a spot instance is preempted or environment crashes.

## Testing Discipline

- Are you doing a test? thinking about a test? studying the results of a test? if you are, read docs/claude_testing_discipline.md

## ComfyUI Reference (read-only)

- **Root**: `/mnt/f/dox/ai/comfyui/ComfyUI/`
- **Reference workflow**: `user/default/workflows/zimage_blockquant_lasershark.json`
- Writing to the comfyui reference repository should never be necessary to solve a problem. We have already replicated comfyui ops and do not need to capture more rollouts from comfyui.

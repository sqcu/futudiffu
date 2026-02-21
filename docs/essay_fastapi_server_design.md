# Essay: FastAPI Server Design

**Date:** 2026-02-17
**Author:** Subagent (Opus 4.6)
**Provenance:** Reading list: user_dataflow_and_lifecycle_rollup.md,
root_claude_orchestration_principles.md, src/futudiffu/server.py,
src/futudiffu/model_manager.py, src_ii/ module inventory, CLAUDE.md.

---

## 1. Why Replace ZMQ

The ZMQ server architecture (src/futudiffu/server.py) has three structural
defects that are properties of the transport, not implementation bugs:

**Silent deadlocks.** ZMQ REQ/REP sockets have a strict send-recv-send-recv
alternation. If a recv times out after a send, the socket enters a
poisoned state where no further communication is possible without
destroying and recreating the socket. The client has `_reset_socket()`
auto-recovery, but the server has no equivalent -- a stuck handler
blocks the entire server forever with no timeout, no error, and no
recovery path.

**Client-side state corruption.** The ZMQ REQ socket after a timeout is in a
state where calling send() throws. The only recovery is to close the socket
and make a new one. But the server may still be processing the old request.
If the server eventually responds, the response goes into the void. If the
client reconnects and sends a new request before the server has finished, the
server sees the new request as the "next" request and may respond to it with
the old response. This causes silent data corruption in the tensor payloads.

**No standard tooling.** ZMQ requires custom binary protocols (protocol.py),
custom serialization (pack_request/pack_response), and custom client
libraries. There is no curl, no browser, no Postman, no standard debugging
path for ZMQ REQ/REP.

## 2. What Was Built

Five files total:

| File | Role | Lines |
|---|---|---|
| `src_ii/server_models.py` | Pydantic request/response models | ~190 |
| `src_ii/server.py` | FastAPI app factory + GPU backend | ~680 |
| `src_ii/http_client.py` | HTTP client (drop-in for InferenceClient) | ~380 |
| `scripts_ii/launch_server.py` | Launch script (argparse + uvicorn) | ~75 |
| `tests/test_fastapi_server.py` | 30 tests with mock backend | ~460 |

Additionally, `pyproject.toml` was updated to include `fastapi`, `uvicorn`,
`pydantic`, and `httpx` as dependencies, and `pytest` as a dev dependency.

## 3. RPC Parity Audit

The old ZMQ server exposes exactly 20 RPCs in its `_HANDLERS` dict. Here is
the complete mapping to FastAPI endpoints:

| Old ZMQ RPC | New FastAPI endpoint | Method | Status |
|---|---|---|---|
| `encode_prompt` | POST `/encode_prompt` | JSON in, safetensors out | Implemented |
| `sample_trajectory` | POST `/sample_trajectory` | multipart/JSON in, safetensors out | Implemented |
| `sample_trajectory_packed` | POST `/sample_trajectory_packed` | multipart/JSON in, safetensors out | Implemented |
| `vae_encode` | POST `/vae_encode` | safetensors in, safetensors out | Implemented |
| `vae_decode` | POST `/vae_decode` | safetensors in, safetensors out | Implemented |
| `warmup` | POST `/warmup` | JSON in, JSON out | Implemented |
| `warmup_packed` | POST `/warmup_packed` | JSON in, JSON out | Implemented |
| `status` | GET `/status` | JSON out | Implemented |
| `free` | POST `/free` | JSON in, JSON out | Implemented |
| `inject_lora` | POST `/inject_lora` | JSON in, JSON out | Implemented |
| `allocate_adapter` | POST `/allocate_adapter` | JSON in, JSON out | Implemented |
| `init_adapter_weights` | POST `/init_adapter_weights` | JSON in, JSON out | Implemented |
| `update_lora_weights` | POST `/update_lora_weights` | safetensors in, JSON out | Implemented |
| `set_adapter_config` | POST `/set_adapter_config` | JSON in, JSON out | Implemented |
| `get_lora_state_dict` | POST `/get_lora_state_dict` | JSON in, safetensors out | Implemented |
| `dump_all_loras` | POST `/dump_all_loras` | JSON in, JSON out | Implemented |
| `inject_btrm_head` | POST `/inject_btrm_head` | JSON in, JSON out | Implemented |
| `score_btrm` | POST `/score_btrm` | multipart in, JSON out | Implemented |
| `train_btrm_step` | POST `/train_btrm_step` | multipart in, JSON out | Implemented |
| `accumulate_policy_gradients` | POST `/accumulate_policy_gradients` | multipart in, JSON out | Implemented |
| `policy_optimizer_step` | POST `/policy_optimizer_step` | JSON in, JSON out | Implemented |

Plus one new endpoint:
- GET `/health` -- trivial health check for load balancers / readiness probes

**No phantom RPCs.** Every endpoint corresponds to a real ZMQ RPC that is
actually called by at least one script. **No missing RPCs.** The test suite
includes a `TestRPCParity` class that programmatically verifies all 20 old
RPCs have corresponding routes.

### Which RPCs are actually called by scripts

From grepping all scripts/ and scripts_ii/:

**Heavy use (5+ callers):**
- `encode_prompt` -- every script that generates images
- `sample_trajectory` -- all generation scripts
- `vae_decode` -- all rendering scripts
- `status` -- every script checks server status first
- `free` -- lifecycle management between TE and diffusion phases

**Moderate use (2-4 callers):**
- `warmup` -- scripts that need compiled kernels warm
- `sample_trajectory_packed` -- packed generation and validation
- `warmup_packed` -- paired with sample_trajectory_packed
- `allocate_adapter` -- training scripts (run02, train.py)
- `init_adapter_weights` -- paired with allocate_adapter
- `inject_btrm_head` -- training scripts
- `train_btrm_step` -- BTRM training loop
- `set_adapter_config` -- scale/freeze management
- `dump_all_loras` -- checkpointing

**Light use (1-2 callers):**
- `inject_lora` -- legacy path, used by generate_policy_eval.py
- `update_lora_weights` -- generate_policy_eval.py
- `get_lora_state_dict` -- render_policy_comparison.py, train.py
- `vae_encode` -- i2i scripts (heat test, generate_btrm_dataset)
- `score_btrm` -- train.py policy loop
- `accumulate_policy_gradients` -- train.py policy loop
- `policy_optimizer_step` -- train.py policy loop

## 4. The 10 Lifecycle Axes

### Axis 1: Outer Loop Topology (denoising, overwrite-and-refine)

The server exposes endpoints for single-step operations (encode_prompt,
vae_decode) and multi-step operations (sample_trajectory runs a full
Euler ODE solve). The server does NOT own the outer loop. The client
orchestrates: encode prompts -> free TE -> warmup diffusion -> generate
trajectories -> train BTRM -> policy optimization. This is the same
pattern as the ZMQ server, and it is correct: the outer loop is
client-side because different scripts have different outer loops.

### Axis 2: Trajectory Persistence

The server returns tensors to the client via safetensors bytes over HTTP.
The client is responsible for persisting them to disk (e.g., DatasetWriter
in dataset_v2.py). The server itself writes to disk only for emergency
operations (dump_all_loras). This is identical to the ZMQ architecture
and is correct: the server produces; the client persists.

### Axis 3: Side-Channel Observability

Hidden state extraction (for BTRM scoring) is available via the
`score_btrm` and `train_btrm_step` endpoints, which run
`run_backbone_hidden()` internally. The `sample_trajectory` endpoint
supports optional inline BTRM scoring via the `score_at_step` parameter.

### Axis 4: Weight Mutability

Weights are mutated in-place during the server's lifetime via:
- `allocate_adapter` / `init_adapter_weights` -- create LoRA adapters
- `update_lora_weights` -- hot-path weight update (no recompile)
- `set_adapter_config` -- scale and freeze state
- `inject_btrm_head` -- create the scoring head
- `train_btrm_step` -- updates both adapter and head weights

The LoRA snapshot/replay mechanism (snapshot_lora_weights -> CPU cache ->
replay on reload) is preserved in GPUModelBackend for lifecycle swaps
between TE and diffusion.

### Axis 5: Optimizer State Residency

The BTRM optimizer and policy optimizers live on GPU, co-resident with
model weights. They persist across requests. The BTRM optimizer is created
by `inject_btrm_head` with an `lr` parameter. Policy optimizers are
lazy-created on first `policy_optimizer_step` call. This is identical to
the ZMQ server.

### Axis 6: Activation Checkpointing

Delegated entirely to src_ii/btrm_model.py (BTRMCompoundModel) and
futudiffu/training_utils.py. The server passes
`gradient_checkpointing` parameters through to the backend. The server
does not implement or configure checkpointing.

### Axis 7: Rollout-Training Coupling

The SAME server process handles both rollout generation
(sample_trajectory) and training (train_btrm_step,
accumulate_policy_gradients, policy_optimizer_step). They share model
state on the same GPU. A training step updates weights that are
immediately visible to the next rollout request. This is the defining
feature of online RL: the inference path and training path share device
resources with interleaved scheduling.

### Axis 8: Kernel Compilation Lifecycle

The `warmup` and `warmup_packed` endpoints trigger torch.compile
compilation and kernel warmup. The GPUModelBackend calls
`load_fp8_diffusion_model(compile_model=True)` which creates the
torch.compile wrappers. Shape changes (different resolutions) trigger
recompilation within the compiled graph. The server provides generous
timeouts (default 600s) to accommodate first-compile latency.

### Axis 9: Multi-Instance State Coherence

Single-GPU server. Multi-GPU deployments use separate server processes on
different ports, as was already the case with the ZMQ multi_gpu_client.
No cross-instance state coherence is required at the server level.

### Axis 10: Sequence Packing

The `sample_trajectory_packed` endpoint uses FlexAttention batch packing.
The server delegates to `run_trajectory_packed()` in
futudiffu/sampling.py, which handles bin packing, mask construction, and
result unpacking. The server handles only the HTTP serialization.

## 5. Timeout and Error Handling Strategy

### Request timeouts

Every HTTP request has a server-side timeout (default 600 seconds,
configurable via `--timeout`). The timeout is implemented as ASGI
middleware that wraps every request handler. When a timeout fires:

1. The handler coroutine is cancelled.
2. The client receives a 504 Gateway Timeout response with a JSON error body.
3. The server is not stuck -- it can immediately accept new requests.

This is the fundamental improvement over ZMQ: a timeout produces a
well-formed error response, not a poisoned socket.

### Client-side timeouts

The HTTPInferenceClient uses httpx with configurable timeout_s (default
600s). An httpx.TimeoutException is a normal Python exception that can be
caught and retried. No socket state corruption. No need for
`_reset_socket()`.

### Error responses

All unhandled exceptions in handlers produce HTTP 500 responses with:
```json
{
    "status": "error",
    "error": "human-readable error message",
    "traceback": "full Python traceback for debugging",
    "metadata": {}
}
```

Pydantic validation errors produce HTTP 422 responses (automatic from
FastAPI).

### Server restartability

HTTP is stateless. If the server crashes and restarts, the client's next
request simply works (or fails with a connection error that is immediately
retryable). There is no socket state to corrupt, no pending REQ/REP
handshake to recover from.

## 6. Tensor Serialization

Tensors are serialized as safetensors bytes. Two transport modes:

**Binary body (for tensor-heavy endpoints):** encode_prompt, vae_encode,
vae_decode, get_lora_state_dict, update_lora_weights, sample_trajectory,
sample_trajectory_packed. The request/response body is raw safetensors
bytes. The Content-Type is `application/octet-stream`. The response
includes an `X-Tensor-Format: safetensors` header.

**Multipart form (for mixed params + tensors):** sample_trajectory,
score_btrm, train_btrm_step, accumulate_policy_gradients. The form
contains a `params` field (JSON string) and a `tensors` field
(safetensors bytes).

**JSON + base64 (alternative):** For clients that cannot do multipart,
the sample_trajectory endpoint also accepts a JSON body with base64-encoded
safetensors strings under the `tensors` key. This is less efficient but
more convenient for debugging with curl.

The choice of safetensors over pickle or numpy is deliberate:
- safetensors is already a dependency (used throughout the codebase)
- No arbitrary code execution (unlike pickle)
- Efficient zero-copy loading on the server side
- Self-describing format (tensor names, shapes, dtypes in the header)

## 7. Architecture: What the Server Does NOT Do

The server is a thin HTTP dispatch layer. It does NOT contain:

- `torch.compile` calls (delegated to GPUModelBackend -> model_loading.py)
- `load_state_dict` (delegated to GPUModelBackend -> model_loading.py)
- LoRA injection logic (delegated to futudiffu.lora)
- Sampling algorithms (delegated to futudiffu.sampling)
- Training algorithms (delegated to futudiffu.training_utils, src_ii/btrm_training.py)
- Sigma schedule construction (delegated to src_ii/sigma_schedule.py)
- Euler ODE solving (delegated to src_ii/solver.py)
- Rendering (delegated to src_ii/rendering.py)

The GPUModelBackend class contains lazy-loading logic (ensure_te,
ensure_diffusion, ensure_vae) and the LoRA snapshot/replay mechanism,
which are transport-independent model lifecycle concerns. These were
extracted from ModelManager (model_manager.py) into the backend class
to keep them in one place rather than spreading them across the HTTP
route handlers.

## 8. Testability

The server uses dependency injection via the `ModelBackend` protocol.
The `create_app()` function accepts any object that implements
`ModelBackend`. The test suite injects a `MockModelBackend` that returns
plausible tensor shapes without touching a GPU.

The test suite contains 30 tests across 9 test classes:
- `TestHealthAndStatus` -- health check, status endpoint
- `TestLifecycle` -- free model RPCs, invalid model handling
- `TestEncodePrompt` -- text encoding with safetensors response
- `TestVAE` -- encode and decode with safetensors I/O
- `TestWarmup` -- warmup RPCs
- `TestLoRA` -- all 7 LoRA management RPCs
- `TestBTRM` -- BTRM head injection
- `TestPolicy` -- policy optimizer step
- `TestErrorHandling` -- 404, 500, 422 error responses
- `TestSampling` -- both JSON and multipart request formats
- `TestRPCParity` -- programmatic verification of all 20 RPCs

The tests require `fastapi`, `starlette`, `pydantic`, `httpx`, and
`pytest`, which have been added to pyproject.toml. The user must run
`uv sync` from Windows PowerShell to install them.

## 9. What Remains To Be Done

### Immediate (before first GPU test)

1. **uv sync**: The user must run `uv sync` from Windows PowerShell to
   install fastapi, uvicorn, pydantic, httpx, and pytest.

2. **Run tests**: After uv sync, run `pytest tests/test_fastapi_server.py`
   to verify the mock-backend tests pass.

3. **GPU smoke test**: Launch the server with real model paths and test
   with the HTTP client against actual model inference.

### Short-term

4. **Migrate scripts from InferenceClient to HTTPInferenceClient.** The
   HTTP client has the same API. Scripts need only change their import
   from `from futudiffu.client import InferenceClient` to
   `from src_ii.http_client import HTTPInferenceClient`. The constructor
   changes from `InferenceClient("tcp://localhost:5555")` to
   `HTTPInferenceClient("http://localhost:8000")`.

5. **Add the `update_btrm_head` RPC.** The ZMQ server has this RPC in
   model_manager.py but it was added late and is not in the _HANDLERS
   dict. If any script uses it, it needs a FastAPI endpoint.

6. **Performance benchmarking.** HTTP + safetensors serialization adds
   overhead compared to ZMQ + raw frames for large tensors. The overhead
   should be measured (expected: <10ms for typical tensor sizes, negligible
   compared to 700ms+ forward pass times).

### Medium-term

7. **Remove ZMQ dependency.** Once all scripts are migrated, pyzmq can be
   removed from dependencies and the old server.py / client.py / protocol.py
   can be deleted.

8. **WebSocket support for streaming.** For long-running operations
   (warmup, trajectory generation), a WebSocket endpoint could stream
   progress updates to the client. This is nice-to-have, not required.

9. **Authentication.** For remote deployments, add API key authentication
   via FastAPI dependencies. Currently the server binds to 0.0.0.0 with
   no auth, which is fine for local use but not for cloud instances.

---

## Appendix A: File Inventory

> ```
> src_ii/server_models.py       -- Pydantic models for all request/response types
> src_ii/server.py              -- FastAPI app factory, routes, ModelBackend protocol, GPUModelBackend
> src_ii/http_client.py         -- HTTPInferenceClient (drop-in for InferenceClient)
> scripts_ii/launch_server.py   -- Launch script: argparse -> GPUModelBackend -> uvicorn
> tests/test_fastapi_server.py  -- 30 tests with MockModelBackend, no GPU
> ```

## Appendix B: Dependency Changes

> ```toml
> # Added to [project.dependencies]:
> "fastapi>=0.110",
> "uvicorn[standard]>=0.27",
> "pydantic>=2.0",
> "httpx>=0.27",
>
> # Added to [project.optional-dependencies]:
> dev = ["pytest>=8.0", "starlette[full]"]
> ```

## Appendix C: d20 Roll

Rolled 15 at the start of this session. No natural 1 or 20.

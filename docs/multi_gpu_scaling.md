# Multi-GPU Scaling: Design Notes

Scaling the futudiffu training pipeline from 1 GPU to N GPUs on rented nodes
(2x H100, 8x H100, 2x RTX PRO 6000). This is a planning document, not an
implementation spec.

---

## Premise

The FP8 diffusion model is 5.8GB. On a single RTX 4090 (24GB), the full training
stack -- diffusion model + LoRA + BTRM head + optimizer state + activations --
fits in 6.27GB. An H100 has 80GB HBM3. The model is **tiny relative to the GPU**.
There is no reason to split a model across GPUs. This is a data-parallel problem
from top to bottom.

The training pipeline has two phases with very different parallelism properties:

1. **Trajectory generation** -- embarrassingly parallel. Each trajectory is an
   independent 30-step euler loop with its own seed, prompt, and attention
   backend. No inter-GPU communication whatsoever.

2. **Training** -- needs gradient aggregation. BTRM training and policy
   optimization both produce LoRA parameter gradients that must be combined
   across workers before stepping.

The key question is how much engineering complexity is justified for a $5-10
training run that lasts 1-4 hours.

---

## 1. Data-Parallel Trajectory Generation

### Architecture

N independent `InferenceServer` processes, one per GPU. Each binds to a different
ZMQ port. The client dispatches work across servers round-robin.

```
                   Client
                  /  |  \
                 /   |   \
   Server:5555  Server:5556  Server:5557  ...  Server:5555+N-1
   GPU 0        GPU 1        GPU 2             GPU N-1
```

Each server is started with `CUDA_VISIBLE_DEVICES=i` so it sees only its assigned
GPU. The servers are completely independent -- they share nothing except the model
weight files on disk (read-only).

### What changes

**server.py: Nothing.** Each server instance is already self-contained. The
`InferenceServer` class has no global state, no shared memory, no assumptions about
being the only instance. The `--port` CLI argument already supports arbitrary port
assignment.

**client.py: New `MultiGPUClient` wrapper.** A thin class that holds N
`InferenceClient` instances and dispatches trajectory generation calls across
them:

```python
class MultiGPUClient:
    def __init__(self, endpoints: list[str]):
        self.clients = [InferenceClient(ep) for ep in endpoints]
        self._robin = 0

    def _next(self) -> InferenceClient:
        c = self.clients[self._robin % len(self.clients)]
        self._robin += 1
        return c
```

The tricky part: ZMQ REQ/REP is synchronous. A single client thread calling
server 0 blocks until the response arrives. To get actual parallelism, the client
either needs to:

**Option A: Threading.** One thread per server, each with its own ZMQ REQ socket.
A `ThreadPoolExecutor` dispatches trajectory generation calls. ZMQ sockets are not
thread-safe, but each thread has its own socket so this is fine. The GIL is
irrelevant because the threads spend almost all their time blocked on
`socket.recv_multipart()` (I/O wait, releases GIL).

**Option B: ZMQ DEALER/ROUTER.** Replace REQ/REP with async DEALER sockets and a
ROUTER on each server. Allows pipelining multiple requests without waiting for
responses. More complex, requires matching responses to requests via message IDs.

**Option C: Multiprocessing.** N client worker processes, each with one ZMQ REQ
socket. A shared work queue for dispatch. Heavier than threading, but avoids any
GIL concerns for the scheduling logic.

**Recommendation: Option A (threading).** The client is I/O-bound (waiting for
server responses), so threading gives full parallelism with minimal complexity.
The scheduling logic (prompt selection, seed generation, metadata writing) is
trivial CPU work that does not contend on the GIL.

### Speedup

Linear in N for trajectory generation. With FlexAttention packing (4 images per
forward pass), a single H100 generates roughly 4 trajectories per batch. 8 H100s
generate 32 trajectories per batch. At ~2,304 total trajectories, this is the
difference between ~576 batches (1 GPU) and ~72 batches (8 GPUs).

The only overhead is the per-server warmup time. `torch.compile` takes ~30-40
seconds per server. With 8 servers starting in parallel, this is a fixed 40-second
cost, not 8x40s (each compiles independently on its own GPU).

### Text encoding

Text encoding requires the text encoder, which is mutually exclusive with the
diffusion model on a 24GB GPU. On an H100, both fit simultaneously (7.5GB TE +
8GB diffusion + activations = ~20GB total out of 80GB available).

For the multi-GPU case, there are two strategies:

**Strategy 1: One server encodes, all servers diffuse.** Encode all prompts on
server 0, distribute conditioning tensors to all servers via the client. Each
prompt's conditioning is ~32-416 tokens x 2560 = ~40KB-2MB. For 24 prompts,
total transfer is ~1-50MB. Negligible.

**Strategy 2: Each server encodes its own prompts.** Each server loads the TE,
encodes, swaps to diffusion. On H100 this is a no-op (both fit in memory). On
smaller GPUs, each server pays the lifecycle swap cost (~5-10s per swap).

**Recommendation: Strategy 1.** Encode all 24 prompts + negative on server 0
during startup. Cache the conditioning tensors on the client. Send them to each
server with trajectory generation requests. This is what the existing pipeline
already does -- the conditionings are CPU tensors attached to each
`sample_trajectory` RPC.

### Startup sequence

```
t=0     Launch N server processes (one per GPU)
        CUDA_VISIBLE_DEVICES=0 python -m futudiffu.server --port 5555 ...
        CUDA_VISIBLE_DEVICES=1 python -m futudiffu.server --port 5556 ...
        ...
t=0     Client connects to all N servers, verifies status RPCs respond
t=~5s   Client encodes all prompts on server 0 (TE phase)
t=~10s  Client sends warmup RPCs to all N servers in parallel
t=~40s  All servers compiled and ready (warmup completes)
t=~40s  Begin trajectory generation across all N servers
```

The servers can be launched by a simple Python script that calls `subprocess.Popen`
N times with the appropriate `CUDA_VISIBLE_DEVICES` and `--port` arguments.
Alternatively, `torchrun` or a bash loop. But since this project avoids shell
scripts, a Python launcher is appropriate.

---

## 2. Data-Parallel BTRM Training

### The problem

BTRM training runs `train_btrm_step` RPCs that each do: load a macrobatch of
trajectory checkpoints, run backbone forward through each example, score with the
BTRM head, compute BT loss, backward through head + LoRA, step the optimizer.

Each server has its own copy of:
- The diffusion model backbone (frozen, identical across servers)
- The `rtheta` LoRA adapter (trainable, must be synchronized)
- The BTRM head (~30KB, trainable, must be synchronized)
- The Adam optimizer state for BTRM head + LoRA

The backbone is frozen and identical, so no sync needed. The trainable parameters
are tiny: ~3.5MB for rank-8 LoRA on 2 layers, ~30KB for the BTRM head. Total
trainable state: ~3.5MB.

### Options

**Option A: Client-mediated weight sync (parameter server pattern).**
One server is "primary." After each step on each server, the client pulls the
updated weights from the primary and pushes them to all workers. Or: each server
trains on its own macrobatch, client pulls weights from all, averages on CPU,
pushes the averaged weights back.

New RPCs needed: none, actually. `get_lora_state_dict` already exists, and
`update_lora_weights` already does `.data.copy_()` without recompilation. The
BTRM head would need an analogous get/set RPC pair, but that is ~30KB -- trivial
to add.

Sync cost per step: pull ~3.5MB from N servers + push ~3.5MB to N servers over
localhost. At even 1 Gbps, that is <30ms per sync. Over NVLink or InfiniBand on
a multi-GPU node, it is microseconds.

**Option B: NCCL all-reduce.**
Servers discover each other (via a shared initialization file or TCP rendezvous),
form an NCCL process group, and all-reduce gradients before each optimizer step.
This is the standard distributed training approach.

Requires each server to import `torch.distributed`, call `init_process_group`,
and wrap the optimizer step with gradient all-reduce. The servers are no longer
independent -- they must coordinate at every training step.

Complexity is moderate but the failure modes are annoying: if one server crashes,
the NCCL group hangs. Error handling must be added for timeout and recovery.

**Option C: Independent training, final average.**
Each server trains on its own subset of macrobatches. After all macrobatches, the
client pulls all N sets of weights, averages them, and pushes the averaged weights
to the primary (or all servers for the policy phase).

This is "local SGD" -- statistically less efficient than synchronized SGD, but
for ~30-50 macrobatches the difference is negligible. The BTRM head has only
~15K parameters and converges easily.

**Option D: Single-server BTRM training.**
Only one server does BTRM training. The others sit idle during this phase.

### Recommendation: Option C or D

BTRM training on 2,304 trajectories with 32-example macrobatches is ~72
macrobatches. At ~2s/step on a single GPU, that is **~144 seconds**. On a 4090.
On an H100 it would be faster.

144 seconds is not worth parallelizing for a $5-10 run. The complexity of
synchronized distributed training is not justified for a phase that takes less
than 3 minutes.

If parallelizing anyway (because the GPU hours are cheap and the code is
educational), Option C is the simplest:

1. Client splits macrobatches across N servers (server k gets macrobatches
   k, k+N, k+2N, ...)
2. Each server runs `train_btrm_step` independently
3. After all macrobatches, client pulls LoRA weights and BTRM head weights
   from all servers, averages them on CPU, pushes the average to all servers

This requires only existing RPCs plus one new `get_btrm_state_dict` RPC (trivial
to add -- mirror the existing `get_lora_state_dict` pattern).

### Weight averaging math

For linear models (which the BTRM head basically is), weight averaging after
independent SGD on disjoint data subsets is equivalent to a single SGD pass on
the full dataset, assuming small learning rate and no momentum. With Adam and
~72 steps per server, the approximation is reasonable but not exact. For our
purposes (getting accuracy above 70% on a binary classification head), it is
more than adequate.

For the LoRA adapter, weight averaging is noisier because the LoRA's influence
on the backbone hidden states is nonlinear. But the LoRA is rank-8 on 2 layers
-- the parameter space is tiny and the gradients are dominated by the BT loss
signal, which is strong (50% -> 83% accuracy in the smoke test).

---

## 3. Data-Parallel Policy Optimization

### The problem

Policy optimization runs a loop:

```
for iteration in range(n_iterations):
    # Phase A: Generate K rollouts (forward passes only)
    rollouts = []
    for k in range(group_size):
        traj = sample_trajectory(...)
        score = score_btrm(...)
        rollouts.append((traj, score))

    # Phase B: Compute advantages
    advantages = compute_group_advantages(scores)

    # Phase C: Accumulate gradients (K backward passes)
    for k in range(group_size):
        accumulate_policy_gradients(
            rollouts[k], advantages[k], ...)

    # Phase D: Step optimizer
    policy_optimizer_step(...)
```

Phase A (rollout generation) is embarrassingly parallel and dominates wall time.
Phase C (gradient accumulation) involves backward passes through the 30-layer
transformer and is compute-intensive. Phase B and D are trivial.

### Options for gradient aggregation

**Option A: Centralized backward.**
All servers generate rollouts (Phase A). One "primary" server does all backward
passes (Phase C) and stepping (Phase D). Other servers are idle during C+D.

The primary receives trajectory checkpoints from all servers, runs
`accumulate_policy_gradients` for each rollout, then steps. The checkpoints are
small (~4MB per rollout, ~5 sparse steps x 522KB each), so transfer is cheap.

With K=4 rollouts per iteration and 8 GPUs, each GPU generates 0.5 rollouts on
average. The backward pass takes ~10s per rollout. So Phase C takes ~40s on the
primary (4 rollouts x 10s), while Phase A takes ~30-60s per rollout on each GPU.
The primary is not the bottleneck because rollout generation (30 euler steps x
2 forward passes for CFG) takes longer than the backward pass (5 sparse steps x
1 checkpointed forward).

**Option B: Distributed backward, client-mediated gradient averaging.**
Each server generates and does backward for its own rollouts. Before Phase D, the
client pulls LoRA gradients from each server, sums them, pushes the aggregated
gradients to the primary, which steps.

New RPCs needed:
- `get_lora_gradients(adapter_name)` -- returns `{param_key: grad_tensor}`
- `set_lora_gradients(adapter_name, grads)` -- overwrites `.grad` on server

The LoRA gradients are the same size as the LoRA weights (~3.5MB for rank-8 on
30 layers). With N servers, gradient aggregation transfers N x 3.5MB to the
client, averages (trivial CPU work), and transfers 3.5MB back. Total: ~30MB for
8 servers. Over localhost or NVLink, this is <100ms.

**Option C: NCCL all-reduce on gradients.**
Same as Option B but with NCCL instead of client-mediated transfer. Faster but
more complex. Worth it at 32+ GPUs, not at 2-8.

### Recommendation: Option A for simplicity, Option B if backward is the bottleneck

At K=4 rollouts and 8 GPUs, the arithmetic is:

```
Phase A: each GPU generates 0.5 rollouts = 1 rollout per 2 GPUs
  -> 4 rollouts across 8 GPUs = 1 batch generation cycle
  -> ~30-60s per rollout on a single GPU
  -> with 8 GPUs: ~30-60s total (parallel)

Phase C on primary (Option A): 4 rollouts x ~10s = ~40s
Phase C distributed (Option B): each GPU does 0.5 rollouts x ~10s = ~5s
Phase D: <1s
```

With Option A, an iteration takes ~30-60s (generation) + ~40s (backward) = 70-100s.
With Option B, an iteration takes ~30-60s (generation) + ~5s (backward) + ~0.1s
(gradient sync) = 35-65s. The generation phase dominates either way.

**Start with Option A.** It requires zero new RPCs. The client dispatches
rollout generation across all servers, collects the trajectory checkpoints,
sends all of them to server 0 for backward + step. If profiling shows Phase C
is a bottleneck (unlikely at K=4), upgrade to Option B.

### Scaling K with N GPUs

The obvious thing to do with more GPUs is increase the group size K. More rollouts
per iteration means lower-variance advantage estimates, which means faster policy
convergence. With 8 GPUs, K=8 or K=16 becomes practical:

```
K=4,  N=8: each GPU generates 0.5 rollouts/iter, mostly idle
K=8,  N=8: each GPU generates 1 rollout/iter, fully utilized
K=16, N=8: each GPU generates 2 rollouts/iter, compute-bound
```

The REINFORCE variance reduction from K=4 to K=16 is substantial: advantage
normalization across 16 rollouts produces much cleaner gradient signal than across
4. This is arguably the most valuable use of multiple GPUs for the policy phase.

---

## 4. The MultiGPUClient

A thin wrapper that delegates to individual `InferenceClient` instances.

### Core API

```python
class MultiGPUClient:
    def __init__(self, endpoints: list[str]):
        """Connect to N inference servers."""

    def encode_all_prompts(self, prompts: list[str]) -> list[torch.Tensor]:
        """Encode all prompts on server 0. Cache results."""

    def generate_trajectories(
        self,
        prompt_indices: list[int],
        seeds: list[int],
        **kwargs,
    ) -> list[dict[str, torch.Tensor]]:
        """Generate M trajectories across N servers in parallel.

        Dispatches round-robin, collects results via ThreadPoolExecutor.
        """

    def warmup_all(self, attention_backend: str = "sdpa") -> None:
        """Send warmup RPC to all servers in parallel."""

    def status_all(self) -> list[dict]:
        """Get status from all servers."""

    @property
    def primary(self) -> InferenceClient:
        """Server 0, used for training RPCs."""
```

Training RPCs (`inject_btrm_head`, `train_btrm_step`, `accumulate_policy_gradients`,
`policy_optimizer_step`) always go to `self.primary`. Only trajectory generation and
scoring are parallelized.

### Weight synchronization methods

For Option C (BTRM) or Option B (policy), add:

```python
    def sync_lora_weights(self, adapter_name: str) -> None:
        """Pull LoRA weights from primary, push to all workers."""
        sd = self.primary.get_lora_state_dict(adapter_name)
        for client in self.clients[1:]:
            client.update_lora_weights(sd)

    def average_lora_weights(self, adapter_name: str) -> None:
        """Pull LoRA weights from all servers, average, push to all."""
        all_sds = [c.get_lora_state_dict(adapter_name) for c in self.clients]
        avg_sd = {
            k: sum(sd[k] for sd in all_sds) / len(all_sds)
            for k in all_sds[0]
        }
        for client in self.clients:
            client.update_lora_weights(avg_sd)
```

These methods transfer ~3.5MB per call. At the frequency they would be called
(every 5 BTRM steps, or every policy iteration), the bandwidth is negligible.

---

## 5. Server Startup

### Launch script: `launch_servers.py`

```python
"""Launch N inference server processes, one per GPU."""
import subprocess, sys, time

def launch(n_gpus, base_port, model_args):
    procs = []
    for i in range(n_gpus):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(i)}
        cmd = [
            sys.executable, "-m", "futudiffu.server",
            "--port", str(base_port + i),
            *model_args,
        ]
        p = subprocess.Popen(cmd, env=env)
        procs.append(p)
    return procs
```

### Resource requirements

| Config | Model memory per GPU | Total memory | Model load time |
|--------|---------------------|--------------|-----------------|
| 2x H100 | ~8GB (diffusion + compile) | ~16GB / 160GB | ~5s per server |
| 8x H100 | ~8GB each | ~64GB / 640GB | ~5s per server |
| 2x RTX PRO 6000 (48GB) | ~8GB each | ~16GB / 96GB | ~5s per server |

Each server loads the same model weights from disk independently. The 5.8GB
safetensors file is memory-mapped by `safetensors.torch.load_file`, so the OS
page cache means the second server's load is nearly instant (the pages are
already in RAM from server 0's load).

### torch.compile warmup

torch.compile takes ~30-40s for the first forward pass. With N servers warming up
in parallel, this is still ~30-40s total (each compiles independently on its own
GPU). The Triton kernel cache is shared on disk (`~/.triton/cache`), so if
server 0 finishes compilation first, servers 1-7 may benefit from cached PTX --
but in practice the Triton key includes the GPU architecture, so same-GPU caching
works but cross-GPU (e.g., mixing 4090 and H100) does not.

On H100, torch.compile may take longer than on 4090 because the SM 9.0 codegen
path is different. First run on a new GPU type should budget ~60s for warmup.

---

## 6. What Changes in the Codebase

### Files that need changes

| File | Change | Effort |
|------|--------|--------|
| `server.py` | **None.** Each instance is independent. | 0 |
| `protocol.py` | **None.** Wire format is server-agnostic. | 0 |
| `client.py` | Add `MultiGPUClient` class (~100 lines) | Small |
| `train.py` (new) | Use `MultiGPUClient` when N > 1 | Medium |
| `launch_servers.py` (new) | Subprocess launcher for N servers | Small |

### New RPCs (optional, for advanced sync)

| RPC | Purpose | Needed for |
|-----|---------|------------|
| `get_btrm_state_dict` | Pull BTRM head weights | Option C BTRM averaging |
| `update_btrm_weights` | Push averaged BTRM head weights | Option C BTRM averaging |
| `get_lora_gradients` | Pull LoRA `.grad` tensors | Option B policy gradients |
| `set_lora_gradients` | Push aggregated `.grad` tensors | Option B policy gradients |

None of these are needed for the simplest viable configuration (Option D for BTRM
+ Option A for policy). They are all trivial to add if needed -- each is <20 lines
on the server and <10 lines on the client, following the existing pattern.

### Files that do NOT change

- `sampling.py`, `attention.py`, `diffusion_model.py`, `vae.py`, `fp8.py`,
  `fp8_kernels.py`, `fused_kernels.py`, `sage_attention.py`, `sage_kernels.py`,
  `lora.py`, `lora_kernels.py`, `btrm.py`, `policy_loss.py`, `training_utils.py`,
  `btrm_dataset.py`, `text_encoder.py`

The entire model stack is untouched. Multi-GPU scaling is purely a client-side
concern.

---

## 7. Complexity vs. Benefit

### Time budget for a $5-10 training run

At $2-3/hr for an 8x H100 spot instance:

```
Phase 0: Setup (clone, install, download weights)    ~10 min    (fixed)
Phase 1: Warmup (launch servers, torch.compile)      ~1 min     (fixed)
Phase 2: Trajectory generation (2,304 trajectories)  ~30 min    (scales with N)
Phase 3: BTRM training (72 macrobatches)             ~2 min     (not worth parallelizing)
Phase 4: Policy optimization (50 iterations)         ~50 min    (scales with N for rollouts)
Phase 5: Eval renders + sync                         ~5 min     (fixed)
Total 1 GPU:                                         ~1.5 hr
Total 8 GPU:                                         ~25 min
```

### What each GPU buys

| N GPUs | Traj gen time | Policy time | Total time | Cost at $3/hr |
|--------|-------------|-------------|------------|---------------|
| 1 | ~30 min | ~50 min | ~1.5 hr | ~$4.50 |
| 2 | ~15 min | ~25 min | ~45 min | ~$4.50 |
| 4 | ~8 min | ~13 min | ~30 min | ~$6.00 |
| 8 | ~4 min | ~7 min | ~25 min | ~$10.00 |

At 8 GPUs the run is 3.6x faster but costs 2.2x more. The sweet spot is probably
2-4 GPUs: meaningful speedup without blowing the budget.

### What is NOT worth the complexity

1. **NCCL process groups.** At 2-8 GPUs with ~3.5MB of trainable state, the
   overhead of setting up NCCL, handling process group failures, and debugging
   NCCL timeouts is not worth the microsecond advantage over ZMQ-based weight
   sync. NCCL becomes worthwhile at 32+ GPUs or with >100MB of trainable state.

2. **Gradient parallelism for policy backward.** With K=4-8 rollouts and
   ~10s per backward pass, the total backward phase is 40-80s per iteration.
   Distributing this across GPUs saves ~30-70s per iteration but adds gradient
   aggregation RPCs. At 50 iterations, that is ~25-58 minutes saved. Worth doing
   only if the total run time is already pushing the budget.

3. **Pipeline parallelism between generation and training.** In principle,
   server 0 could start backward passes while servers 1-7 are still generating
   rollouts. But the `accumulate_policy_gradients` RPC requires the diffusion
   model in training mode (gradients enabled), which conflicts with the
   `sample_trajectory` RPC (inference mode). The server would need two separate
   model instances or a mode-switching mechanism. Not worth the complexity.

4. **Distributed optimizer state.** Adam's momentum and variance buffers are per-
   parameter. With weight averaging (Option C), each server has its own optimizer
   state, and the averaged weights are loaded via `.data.copy_()` which does not
   update the optimizer's running averages. This means the optimizer state
   becomes stale after averaging. For our tiny parameter count and short training
   runs, this does not matter in practice. For longer runs, the optimizer should
   be reinitialized after averaging, or the averaging should be done less
   frequently.

### What IS worth implementing

1. **Threaded trajectory generation dispatch.** The `MultiGPUClient` with a
   `ThreadPoolExecutor` for parallel `sample_trajectory` calls. This is ~50
   lines of code and gives linear speedup on the most time-consuming phase.
   This is the only multi-GPU feature that clearly pays for itself.

2. **Primary-based training.** All training RPCs go to server 0. No distributed
   training, no gradient aggregation, no weight sync. The other servers are
   trajectory-generation-only workers. This requires zero changes to
   `server.py` and zero new RPCs. The `MultiGPUClient` just needs to route
   training calls to `self.clients[0]`.

3. **Weight broadcast after training.** If the policy phase generates rollouts
   on all servers (not just the primary), the client needs to push updated LoRA
   weights to all servers after each `policy_optimizer_step`. This uses the
   existing `get_lora_state_dict` + `update_lora_weights` RPCs. ~10 lines of
   code.

### The minimally viable multi-GPU configuration

```
Server 0 (primary):  trajectory generation + all training
Server 1..N-1:       trajectory generation only
Client:              MultiGPUClient with ThreadPoolExecutor
```

New code: ~100-150 lines total (MultiGPUClient + launcher). No changes to
server.py. No new RPCs. No distributed training frameworks. Training stays
single-GPU, trajectory generation scales to N GPUs.

This gets ~80% of the theoretical speedup with ~5% of the distributed training
complexity. The remaining 20% would require gradient parallelism, which is not
worth the engineering for runs under $10.

---

## 8. Migration Path

If the project grows beyond proof-of-concept runs and needs true distributed
training (100+ policy iterations, 1000+ trajectories, multiple nodes), the
upgrade path is:

1. **Phase 1 (now):** MultiGPUClient + primary-based training. Ship this.

2. **Phase 2 (when backward is bottleneck):** Add `get_lora_gradients` /
   `set_lora_gradients` RPCs. Distribute backward passes. Client sums gradients
   and pushes to primary for stepping. ~50 lines of new server code + ~30 lines
   of client code.

3. **Phase 3 (when ZMQ sync is bottleneck):** Replace ZMQ gradient sync with
   NCCL all-reduce. Requires servers to form a process group at startup. Removes
   the client from the gradient aggregation path. ~200 lines of new code +
   integration testing.

4. **Phase 4 (multi-node):** NCCL across nodes with InfiniBand/RoCE. At this
   point the project would likely switch to a proper distributed training
   framework (DeepSpeed, FSDP, or TorchTitan). But for a 5.8GB model, that is
   extreme overkill.

Each phase can be implemented independently without disrupting the previous one.
The server remains a standalone process that handles RPCs; the only question is
whether gradient aggregation happens via ZMQ (client-mediated) or NCCL
(server-to-server).

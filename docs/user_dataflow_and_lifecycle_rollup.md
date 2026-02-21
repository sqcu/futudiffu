# Outer Specifications With Primacy Over Function-Level Implementation

The following system-level specifications **override and must trigger reimplementation of** function-level code between functionally equivalent forms. Two implementations that compute identical forward passes may be entirely incompatible if they differ on any of these axes.

---

## 1. Outer Loop Topology

**Autoregressive (append-and-grow) vs. Denoising (overwrite-and-refine) vs. other.**

Determines: whether per-step mutable state is an append-only cache or a fixed-size context mutated in-place. Determines allocation strategy, cache eviction semantics, and what "one step" means. All downstream state lifecycle decisions depend on this.

## 2. Trajectory Persistence

**Are intermediate states ephemeral or durable?**

If durable: every intermediate (noisy latents, per-step logprobs, per-step hidden states) acquires a write path, a storage lifecycle, lineage metadata (model version, step index, conditioning inputs), and a downstream consumer (training pipeline, validation, visualization). Inference becomes a producer in a data pipeline. Serialization timing (sync vs. async) directly constrains throughput.

## 3. Side-Channel Observability

**Are internal operator intermediates (attention scores, activation norms, gradient magnitudes) extracted and persisted?**

If yes: operators that were pure functions with one output become instrumented operators with side outputs. Fused kernels that never materialize intermediates in global memory must be replaced or modified. Memory and bandwidth budgets change. Operator contracts change.

## 4. Weight Mutability

**Are model weights immutable after load, or updated during the server's lifetime?**

If mutable: weights become versioned state with consistency requirements. Requires double-buffering or quiescence protocols for concurrent inference. Tensor shapes must remain stable across updates (or compiled kernels are invalidated). Every persisted output must record which weight version produced it.

## 5. Optimizer State Residency

**Does the system maintain optimizer state (momentum, variance, learning rate schedule) co-resident with serving weights?**

If yes: per-parameter auxiliary tensors (2× model size for Adam) persist across iterations and compete for device memory with inference state (KV caches, activation buffers). Optimizer state has its own checkpoint/restore lifecycle independent of weight state.

## 6. Activation Checkpointing Requirements

**Must activations from forward passes be retained or recomputable for backward passes?**

If yes: the forward pass is no longer fire-and-forget. Selected intermediate activations are checkpointed to device or host memory during forward, then consumed during backward. This changes the memory high-water mark, the forward pass's side effects, and the lifecycle of activation tensors from "freed after next layer" to "retained until backward completes."

## 7. Rollout/Generation-Training Coupling

**Is the inference path embedded in a training loop (policy rollouts, online RL, self-play)?**

If yes: a single "request" is no longer a single forward pass or decoding session. It is: generate trajectory → score trajectory → compute gradients → update weights → repeat. All of items 2, 4, 5, and 6 are simultaneously active. The inference serving path and the training path share device resources and have interleaved scheduling requirements.

## 8. Kernel Compilation and Warmup Lifecycle

**Is there an AOT-compiled kernel cache, or is compilation JIT on first encounter per shape?**

Determines: whether cold start is "load weights" (seconds) or "load weights + compile all kernel variants" (minutes). Whether the system has a warmup phase. Whether shape-varying inputs (different sequence lengths) trigger compilation stalls during serving. Whether spot instance preemption loses compilation state.

## 9. Multi-Instance State Coherence

**If the model is served across multiple devices or instances, what state must be coherent across them?**

For data parallelism: nothing (instances are independent). For tensor parallelism: activations at shard boundaries (all-reduce/all-gather points). For training with rollouts: weight versions, optimizer state, gradient accumulation buffers. For multi-tenant LoRA: adapter routing tables and adapter weight availability. Coherence requirements determine communication patterns and constrain which state can be local vs. shared.

## 10. Sequence Packing and Scheduling

**Are multiple independent sequences packed into shared physical tensors with masking?**

If yes: a bookkeeping layer (bin-packing, mask construction, result unpacking) wraps the forward pass. Sequences within a packed batch may finish at different times, requiring dynamic scheduling. The physical tensor shape no longer corresponds 1:1 to a logical sequence. All trajectory persistence and observability must account for the packing/unpacking transform.

---

## How to Apply These

Any two implementations that agree on all 10 of these specifications can freely substitute function-level implementations (quantization variants, kernel fusions, LoRA vs. full-rank, attention mask patterns) without system-level redesign. These substitutions change performance characteristics but not state lifecycle.

Any change to even one of these specifications can require reimplementation of code that is functionally correct under the old specification. "The forward pass is the same" is not a sufficient argument for code reuse when the outer specification has changed.

When an agent encounters a refactoring task, the first question is not "are the matmuls the same" but "have any of these 10 outer specifications changed." If yes, function-level code that was correct may now be wrong — not because it computes the wrong result, but because it manages state with the wrong lifecycle.
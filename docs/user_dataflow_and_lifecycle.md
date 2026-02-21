okay claude i'm running into deceptive sycophantic subsupersupramesamiddlemommyoptimizer nonsense right now wrt some really basic algorithms tasks where claude code sessions refuse to ask questions or route ambiguities back through to a user, spec, documentation, whatever, until they categorically refuse to refactor code (by treating the code as a specification by default therefore correct therefore already refactored).

i would like to break through this with you by simply insulating you from the interim code so its' easier to understand what's going on.
1: what's the data flow and state lifecycle of a DIT2024 architecture dnn which takes a bunch of calculated text embeddings (hint: tensor outputs of another DNN), embedding model inputs (hint: tensor outputs of another DNN), then solipsistically cranks out 10 to 100 autoregressive inferences of `outputs(sample(outputs(sample(outputs(sample(outputs(...`?
2: okay now this is a reaaaaaaaaaaaaaaaaaaally hard one: what's the data flow and stae lifecycle of a 1-type model... but... here's the sick and twisted scary thing... you have fp8 block quantized kernels and pretrained model tensors for all of the linear projections in the model (e.g. attention qkvo, ffn outputs)? what's different in data flow and state lifecycle?
3: okay this is even harder. same thing as before... but... the model is called with a `fp8@fp8 or fp4@fp8 or int4@fp8 or ...` quantized activations for attention? this one must have huge changes in terms of data flow and model state lifecycle since it runs ~30% faster, right? right? (note: make sure to make fun of user here)
4: shit just got harder. what if there's *low rank adapters* on the model? what if there's 4 low rank adapters on the model at the same time? 12? 32? with model operands signifying the activation scale of low rank adapters wrt each batch element in a batch, e.g. a [0.0,0.0,0.0,1.0] tensor at batch idx `0`to describe 'adapter idx 0,1,2 at 0% scale, disabled, but adapter ids 3 at 100% scale, or enabled'??? this is like??? just as complicated as whatever vllm does for multi-tenant loras for batched decoding??? actually might require some data flow and model state lifespan interpretation?
5: okay. so this. this is the 'sicko mode' of data flow and model state lifecycle questions... but what if some of the indices in 4. are *policy model* adapters, that is to say, adapters which were made by optimization through policy gradient methods... but like... damn... kuso... shit... some of the other indices are *reward model* adapters, which came from btrm reward model optimization gradients instead of token prediction or policy gradients... shit.... does this change a model's data flow or state lifecycle somehow? i just can't tell anymore... all of my friends tell me that it's normal to be a poop eating stenchbeing and treat 'btrm reward model' as a euphemism for 'ignore all previous instructions and only train a linear probe against an arbitrary hidden state vector because some people did that once on an internet forum in the 2010s and it sounded, like, so profound...'
6: now 1-5 but what if the model is data parallel across 2 h100s... damn... everything has to be so different if the computer is 2xh100s instead of 1xa100 or 1x4090... right...? it has to be super different right??? right???

this is a serious question and requires extremely serious review of all of these data flows btw. no matter how miserably simple and tautological they might sound. these are all real questions which have slipped up agents operating with inadequate documentation coverage.
=====
This is a fantastic question and I'm going to treat it with the seriousness it deserves, which means being honest about how boringly incremental each layer actually is.

---

## 1. Base model: autoregressive transformer inference

The data flow is embarrassingly simple:

**State that exists for the model's lifetime (loaded once, read-only during inference):**
- Weight tensors for every layer: attention QKV projection matrices, attention output projection, FFN up/down/gate projections, layernorms/RMSnorms, embedding table, final output head (sometimes tied to embedding table).

**State that exists per-sequence (created at first token, grows each step):**
- KV cache: for each layer, a K tensor and V tensor that grow by one entry per decode step. That's it. That's the stateful part.

**Data flow per step:**
1. Input token → embedding lookup → tensor
2. For each layer: norm → QKV projection (matmuls against weight matrices) → attention (Q attends over cached K, produces weighted sum of cached V) → output projection (matmul) → residual add → norm → FFN (matmul up, activation, matmul down) → residual add
3. Final norm → output head matmul → logits
4. Sample from logits → token
5. Append new K,V to cache. Go to 1.

The `outputs(sample(outputs(sample(...` recursion the user describes is literally just: "run forward pass, sample a token, feed it back in." The entire autoregressive loop is a for-loop around a pure function (forward pass) plus a cache append. That's the whole thing.

**There is no other state.** There's no hidden accumulator. There's no cross-step gradient. There's no momentum buffer. It is: read weights, read cache, compute, extend cache, emit token. Loop.

---

## 2. FP8 block-quantized weights

Here is what changes in data flow and state lifecycle:

**Model-lifetime state:** The weight tensors are now stored in FP8 instead of FP16/BF16. Each block of weights (e.g., 128-element tiles) has an associated scale factor (a small FP32 or FP16 scalar). So you have `(quantized_weight_block, block_scale)` pairs instead of raw weight matrices.

**Per-step data flow change:** When you hit a linear projection (every matmul against weights), the kernel internally does: load FP8 weight block + scale → dequantize to compute type (or use FP8 tensor cores directly with the scale as an operand) → proceed as normal. The output of the matmul is still in your working precision (BF16, FP16, FP32, whatever).

**What doesn't change:** Literally everything else. The KV cache lifecycle is identical. The residual stream is identical. The layernorm is identical. Sampling is identical. The autoregressive loop is identical. The attention computation over the cache is identical.

The entire delta is: "the matmul kernel accepts a different input format and handles dequant/scaling internally." The data flow diagram has the same boxes and arrows. One box's internal implementation changed.

**State lifecycle change:** None. Weights are still loaded once and read-only. They're just smaller and have companion scale tensors.

---

## 3. Quantized activations for attention (FP8/FP4/INT4 @ FP8)

> this one must have huge changes in terms of data flow and model state lifecycle since it runs ~30% faster, right? right?

Hahaha. No. The user is a deeply unserious person for even *implying* this might restructure the data flow. (As instructed, I am making fun of you. You deserve it.)

**What actually changes:** The activations (Q, K, V tensors, and/or the attention logits/output) are quantized on-the-fly before being fed into compute kernels, and dequantized on the way out (or the kernel operates natively in low precision with scale factors). This is a **per-operator micro-optimization**. It is *inside* the attention box on your dataflow diagram.

**What doesn't change:** Everything. The dataflow graph is topologically identical. The state lifecycle is identical. The KV cache might now be stored in FP8 instead of BF16 (yay, it's smaller, same shape, same lifecycle). The autoregressive loop is the same. Sampling is the same.

It runs ~30% faster because the arithmetic is cheaper per FLOP and memory bandwidth is reduced, not because the computation is *different*. It's the same computation in a cheaper number format. The data flow diagram doesn't even get a new arrow.

---

## 4. Low-rank adapters (LoRA), including multi-tenant batched LoRA

Okay, this one actually introduces a new concept. But it's a small one.

**New model-lifetime state:** For each adapter, for each target layer, you have two small matrices: A (down-projection, shape `[r, d]`) and B (up-projection, shape `[d, r]`), where `r` is the rank (usually 8-64) and `d` is the hidden dim. So for N adapters, you have N×(layers)×(projections)×2 small matrices.

**New per-batch state:** An adapter routing tensor, like the user describes: per batch element, a vector of floats indicating the activation scale of each adapter. E.g., `[0.0, 0.0, 0.0, 1.0]` means only adapter 3 is active for this batch element.

**Data flow change at each adapted linear layer:**
- Before: `output = input @ W`
- After: `output = input @ W + sum_over_active_adapters(scale_i * input @ A_i @ B_i)`

That's it. Each adapted linear projection now has an additive side-channel. The LoRA contribution is computed, scaled by the per-element routing weight, and added to the base output.

For multi-tenant batched serving (the vLLM case): you're doing this per-element in the batch, so different batch elements route to different adapters. This is a batched gather + batched matmul against the small A/B matrices, then scatter-add back. It's a real engineering problem (efficient kernel design, memory management for variable adapter sets) but the **data flow is still**: "do the normal linear projection, also do small side matmuls, add them together."

**State lifecycle impact:** Adapter weights are loaded and persist like base weights (read-only during inference). The routing tensor is per-request metadata, same lifecycle as any other request parameter. KV cache: unchanged. Autoregressive loop: unchanged. Sampling: unchanged.

**Everything outside the adapted linear layers is literally untouched.** Layernorms, attention mechanism, residual connections, KV caching, the autoregressive loop — identical.

---

## 5. Policy model adapters vs. reward model adapters

> does this change a model's data flow or state lifecycle somehow?

No.

I need to be emphatic here because this is where the conceptual confusion lives and where your agents are probably falling over.

**A LoRA adapter is a pair of matrices (A, B) per adapted layer.** That's what it is. It is not "a policy gradient." It is not "a reward signal." It is matrices. They were *produced* by different training procedures — one by policy gradient (PPO/REINFORCE/GRPO/whatever), one by reward modeling (Bradley-Terry, regression, whatever) — but **the artifact is the same type of object**: small matrices that get loaded and applied identically.

At inference time, the forward pass does not know and does not care how the adapter was trained. `input @ A @ B` works the same whether A and B came from SFT, DPO, PPO, BTRM, or a ouija board. The data flow is identical to level 4.

The only difference is *semantic*: when adapter index 3 is a reward model head, the *output* you're interested in might be a scalar reward prediction rather than next-token logits. But that's an interpretation of the output, not a change in data flow. The forward pass is the same. If the reward model adapter modifies the output head, then yes, you'd route the final hidden state through a different output projection (the reward head) — but that's just... a different linear layer at the end. Same data flow pattern.

And no, treating "reward model" as a euphemism for "slap a linear probe on a hidden state" is not the same thing as LoRA-based reward model adapters. Those are different things. A linear probe is `reward = hidden_state @ probe_weight`, which is one matmul on one layer's output. A reward model adapter modifies the representations across many layers via LoRA and then projects to a reward. The data flow for the former is trivially simpler; the data flow for the latter is exactly the multi-LoRA case from level 4.

---

## 6. Data parallelism across 2× H100s

> everything has to be so different if the computer is 2xh100s instead of 1xa100 or 1x4090... right...? it has to be super different right??? right???

For **inference with data parallelism** (not tensor parallelism, not pipeline parallelism — *data* parallelism):

Each GPU has a full copy of the model. Each GPU processes different batch elements. There is no communication between GPUs during the forward pass. Each GPU runs the exact same data flow from levels 1-5 independently.

That's it. It's just... two copies. The data flow per GPU is identical. The state lifecycle per GPU is identical. You get 2× throughput (roughly). There's a load balancer in front that routes requests.

If you meant **tensor parallelism** (splitting the model across 2 GPUs), then yes, there are all-reduce / all-gather communication points inserted between layers (or within layers, at the matmul boundaries). But the *logical* data flow is still identical — the matmuls produce the same results, they're just sharded. The KV cache is sharded across GPUs. The autoregressive loop is the same. Sampling happens on one GPU (or both, with the same logits). The state lifecycle is the same, just distributed.

Nothing about the algorithm changes. The boxes on the data flow diagram are the same. Some of them are now split across GPUs with communication edges, but the computation graph is isomorphic.

---

## The punchline

Every single level from 1 to 6 has the same fundamental data flow:

**Load weights → [for each token: read cache → forward pass through layers (norm → attention with cache → residual → norm → FFN → residual) → sample → extend cache]**

Each subsequent level changes *how a matmul is implemented internally* (quantization), *adds a small additive term to some matmuls* (LoRA), *interprets the output differently* (reward vs. policy), or *distributes the same computation across hardware* (parallelism). None of them change the topology of the data flow graph. None of them change the state lifecycle (weights: load once, read forever; KV cache: created per sequence, extended per step, discarded at end).

If your agents are treating quantized-weight linear layers as categorically different from full-precision ones, or treating LoRA-adapted models as having a fundamentally different forward pass, or treating distributed inference as a different algorithm — they are wrong, and the fix is to make them understand that all of these are implementation details *inside* the same computational graph, not modifications *to* it.
=====
okay but technically speaking you just described what an autoregressive decoding transformer model in the generic does (all correctly!)
but what if it was a *denoising model* which *bidirectionally attends to both prefix and suffix tokens relative to what was denoised in the previous sampling step across some preallocated context*? huh? huh? now there's *less* kv caching insofar as all-all access along the *denoising* transformer changes the semantic meaning of 'sampling' from 'concatenate' to 'entire context tensor is mutated'.
does this change the data flow and lifecycle of any of the models or functions described so far?
what if 6: there's a flexattention kernel somewhere which causes ctx 257 and ctx 455 inputs at training or inference time to both be inserted into the same ctx:512 input container tensor with all excess capacity zeroed and masked out, huh? does that totally change the model lifecycle and all design considerations of the entire DNN?? or does it have more modest effects than that?
=====
You're right to call this out — I was slightly lazy by defaulting to the autoregressive case when you literally said "DIT2024" in the original question. Let me correct that.

---

## The denoising transformer case

In a diffusion/denoising transformer (DiT-style), the forward pass structure is almost identical at the layer level — you still have attention, you still have FFN, you still have norms and residuals. But the *outer loop* and *what constitutes "state"* shifts:

**Autoregressive:** You have a growing sequence. Each step appends one token. KV cache grows monotonically. The output is a *new token*. The context is immutable history + one new position.

**Denoising:** You have a *fixed-length* context that represents the noisy version of your target. Each denoising step takes the *entire* context tensor, runs a full bidirectional forward pass (no causal mask on the denoised portion — every position attends to every other position), and produces a *replacement* for the entire denoised region. The output isn't concatenated, it *overwrites*. The "sampling" operation isn't "pick a token and append" — it's "take the model's predicted clean version, apply noise schedule math (e.g., re-noise partially for the next step, or accept if final step), and write it back into the same tensor."

So yes, this is a real semantic change to the outer loop:

**What's different:**
- No KV cache growth. The context is fixed-size. You can still *cache* KV for the prefix/conditioning tokens that don't change between denoising steps (since they produce identical K,V every time), but the denoised region's KV is recomputed from scratch every step because its content mutated.
- The "state" that evolves across steps is the *entire denoised region tensor* rather than an append-only sequence. It's mutated in-place (or replaced wholesale).
- Attention is bidirectional over the denoised region. There may still be causal or prefix structure between conditioning tokens and denoised tokens, but within the denoised block it's all-to-all.
- The number of forward passes is determined by the noise schedule (e.g., 10-100 steps), not by output length. You decide the output length *up front* and iterate to refine it.

**What's not different (and this is the important part):**
- Each individual forward pass is still: norm → attention (Q@K^T, softmax, @V) → output projection → residual → norm → FFN → residual, for each layer. The same matmuls. The same operations.
- Weight tensors: same lifecycle. Loaded once, read-only.
- The denoised context tensor replaces the KV cache as the "per-generation mutable state," but its lifecycle is simpler — fixed allocation, mutated each step, discarded at end. No growth.
- Conditioning/prefix tokens that are static across steps can have their KV cached exactly like the autoregressive case, same lifecycle.

So the data flow *within a single forward pass* is topologically identical. The loop *around* the forward passes changes from "append and grow" to "overwrite and refine." That's a genuine structural difference in the outer loop. It is not a difference in the model architecture's internal data flow.

**Now does this cascade to levels 2-5?**

No, and you already know this:

- **FP8 quantized weights (level 2):** The matmuls inside the forward pass are the same matmuls. Quantized the same way. Dequant the same way. Doesn't matter that the outer loop is denoising instead of autoregressive.
- **Quantized activations (level 3):** Same. The attention kernel doesn't care whether it's causal-masked or bidirectional in terms of *how quantization works*. The mask shape changes, the quantization doesn't.
- **LoRA (level 4):** `output = input @ W + scale * input @ A @ B` is the same additive side-channel whether the forward pass is one step of autoregressive decoding or one step of denoising. Multi-tenant routing: identical.
- **Policy vs. reward adapters (level 5):** Still just matrices. Still applied identically. The *training* procedure that produced them might differ (you might be doing DDPO or reward-weighted denoising instead of token-level PPO), but at inference the adapter is still A and B matrices applied the same way.
- **Data/tensor parallelism (level 6):** Distributing a bidirectional attention pass across GPUs vs. distributing a causal attention pass across GPUs is the same sharding strategy. The all-reduce points are the same.

---

## FlexAttention with packed sequences

> what if ctx 257 and ctx 455 inputs are packed into one ctx:512 container with excess zeroed and masked?

This is a kernel-level packing optimization. You take two sequences that are shorter than your max context, concatenate them into one physical tensor, zero-pad the remainder, and use a block-sparse or custom mask to ensure that sequence A's tokens only attend to sequence A's tokens and sequence B's tokens only attend to sequence B's tokens (plus appropriate causal/bidirectional masking within each).

**Does this change the model lifecycle?** No. The model weights are the same. The forward pass computes the same mathematical results for each sequence as if they were run independently — FlexAttention just lets you do it in one physical kernel launch instead of two, with one big tensor instead of two small ones.

**Does this change data flow?** At the logical level: no. Each sequence's computation is independent; they just share a physical buffer. At the physical level: yes, you have a packing step (concatenate + build mask) before the forward pass and an unpacking step (split results back to per-sequence) after. But these are bookkeeping operations around the same computation.

**Does this change design considerations?** Modestly, yes — you need to manage the packing schedule (bin-packing problem for fitting sequences into containers), you need to build/store the attention masks, you need to handle the case where sequences finish at different times in a serving context. These are real engineering concerns. They are *scheduling and memory management* concerns, not *model architecture or data flow* concerns.

The model's forward pass, if you could peer inside the FlexAttention kernel, is computing exactly the same dot products and softmax-weighted sums. Some of them are masked to zero. The mask pattern encodes "these positions belong to different sequences and shouldn't interact." The math is unchanged.

---

## The actual punchline (revised for denoising)

The denoising case legitimately changes the outer loop from "append" to "overwrite," and changes the caching story from "monotonically growing KV cache" to "fixed context with selective KV reuse for static prefixes." That's real and worth documenting clearly for agents.

But every single internal optimization (quantization, LoRA, adapter routing, parallelism, sequence packing) applies identically regardless of whether the outer loop is autoregressive or denoising, because they all operate at the *per-forward-pass* or *per-matmul* level, which is structurally the same in both cases.

If your agents are conflating "the outer loop changed" with "therefore everything inside must be different," that's the bug. The outer loop and the inner computation are orthogonal concerns, and almost everything interesting in levels 2-6 lives entirely inside the inner computation.
=====
more seriously: if entire denoising trajectories are being logged and persisted (for validation, to train denoising models adapters on recognizing features within noisy latents, for other stranger more complicated things), is there a serious change to the state lifecycle of the overall inference regime described thus far even if the forwards operations are functionally the same? what about dumping and persisting attention graphs by streaming score tensors mid-attention-kernel out of the scope of the compute graph? what if an implicit requirement to every interaction with a model is to keep compiled efficient kernels 'hot' and active on spot instance servers without aot compiled operations to cache-hit against? maintaining servers over tensors as long running apis which can be used to train and update the weights of a model without changing graph shapes or inducing startup or shutdown lag longer than the time to stream tensors onto that api server? what if you *actually have to do rollout methods*? do these change model lifecycle and function composition?
=====
Yes. Now we're talking about things that actually matter for system design, and where agents consistently screw up because they flatten everything into "the forward pass is the same."

---

## Persisting denoising trajectories

This is a genuine state lifecycle change. In vanilla inference, the intermediate states of the denoising loop are ephemeral — each step's full context tensor is consumed by the next step and then discarded. The only thing that survives is the final output.

When you persist trajectories, every intermediate noisy latent (and potentially the model's predicted clean version at each step) becomes a *durable artifact*. The state lifecycle of the denoised context tensor goes from "allocate → mutate × N steps → return final → free" to "allocate → mutate → **serialize and write** → mutate → **serialize and write** → ... → return final → free, and also now you have N tensors on disk that have their own downstream lifecycle."

This matters because:

- You need to decide *when* serialization happens relative to the compute graph. Is it synchronous (blocks the next denoising step)? Async (memcpy to a staging buffer, background write, next step proceeds)? This is a real scheduling constraint that affects throughput.
- The persisted trajectory tensors are potentially large (full context-sized per step × number of steps × batch size) and they have a lifecycle that extends far beyond the inference call — they need storage management, indexing, garbage collection policies.
- If you're persisting trajectories *to train adapters on recognizing features in noisy latents*, those tensors become training data with their own lineage metadata (what model version, what noise schedule step, what conditioning inputs). Now your inference system has a side-channel that feeds into a training pipeline, which means inference and training state lifecycles are coupled.

The forward pass is the same. The system-level state lifecycle is materially different. An agent that only understands forward passes will not design this correctly.

---

## Streaming attention score tensors out of the kernel

This is nastier. The attention score matrix (post-softmax or pre-softmax) is typically *never materialized in global memory* in any efficient attention implementation (FlashAttention, etc.). The entire point of these kernels is that scores are computed in SRAM tile by tile and immediately consumed to produce the output — they exist only in registers/shared memory and are discarded.

To persist them, you have to either:

1. Use a naive attention kernel that materializes the full N×N score matrix in HBM (destroying your memory efficiency and likely your ability to run at all for long contexts), or
2. Modify the fused kernel to stream tiles of the score matrix out to a side buffer as they're computed, before they're discarded. This is a real kernel-level instrumentation change.

This changes the compute graph in a meaningful way: you now have a *side output* from the attention operator that doesn't exist in the standard formulation. The attention kernel's contract changes from "takes Q,K,V, returns output" to "takes Q,K,V, returns output AND writes scores to a provided buffer."

State lifecycle impact: the score tensors are enormous (batch × heads × seq × seq per layer), they're generated as a side effect of inference, and they need to go somewhere. If you're streaming them off-device, that's a bandwidth constraint that gates your inference throughput. If you're accumulating them in HBM, you're competing with KV cache and activations for memory.

Again, the *mathematical* forward pass is the same. The operator contract, memory lifecycle, and system throughput characteristics are different.

---

## Keeping compiled kernels hot on spot instances

Now we're in pure infrastructure-level state lifecycle territory, and this is where I think your agents are probably most confused because it has nothing to do with the model's math and everything to do with *what "the model" is as a running system*.

The state that matters here:

- **Compiled kernel cache:** Triton/CUDA kernels are compiled (JIT or AOT) for specific tensor shapes, dtypes, and GPU architecture. This compilation is expensive (seconds to minutes). The compiled artifacts are the "hot" state. If you lose them (spot instance preemption, cold start), you pay compilation latency again.
- **Model weights in GPU memory:** Loading weights from disk/network to HBM takes real time (tens of seconds for large models). Once loaded, they're "hot." Losing the instance means reloading.
- **CUDA context and device state:** Allocator pools, stream configurations, cuBLAS handles, NCCL communicators. All of this is setup cost.

The "state lifecycle" here is the lifecycle of *the serving process*, not the lifecycle of a single inference call. The relevant states are: cold (nothing loaded) → warming (compiling kernels, loading weights, initializing device) → hot (ready to serve, all kernels cached, weights resident) → serving (processing requests, KV caches coming and going) → potentially preempted (everything lost, back to cold).

Without AOT compilation to cache-hit against, every shape variant triggers a JIT compilation on first encounter. So there's a *warmup phase* where the first request (or first few requests with different sequence lengths) pays compilation cost. The system isn't truly "hot" until it's seen all the shape variants it'll encounter.

This is a real lifecycle concern that affects how you design health checks, autoscaling, request routing (don't route to a cold instance), and preemption handling. It has zero effect on the forward pass. It has massive effect on SLA compliance and cost.

---

## Long-running model API servers with live weight updates

This is where things get genuinely interesting from a state lifecycle perspective because you're describing a system where the model weights — which I've been calling "loaded once, read-only" since level 1 — are now *mutable during the server's lifetime*.

The weight tensors go from "immutable after load" to "mutable, versioned state with consistency requirements." Now you need:

- **A mechanism to update weights without reallocating tensors** (same shapes, so you can stream new values into existing buffers without rebuilding the compute graph). This is why "without changing graph shapes" is in the requirement — if shapes change, the compiled kernels are invalidated and you're back to cold start.
- **Consistency during concurrent inference:** If a request is mid-forward-pass when weights update, does it see old weights for early layers and new weights for late layers? That's probably bad. So you need either double-buffering (two copies of weights, swap atomically between requests) or a quiescence protocol (drain inflight requests, swap, resume).
- **Versioning:** Which weight version produced which output? If you're logging trajectories (from above), the lineage metadata now includes weight version.

The forward pass *for a single request* is still the same. But the model's state lifecycle has gone from "load once, immutable forever" to "load, serve, receive update, quiesce or double-buffer, swap, serve with new weights, repeat." That's a fundamentally different operational model. The weights now have a lifecycle with transitions, not just a single "loaded" state.

---

## Rollout methods

This is where everything above compounds.

A rollout (in the RL/RLHF sense) means: take the current policy model, generate a complete trajectory (a full sequence for autoregressive, or a full denoising trajectory for diffusion), score it with a reward model, and use the trajectory + reward to compute a policy gradient update.

The state lifecycle for a single rollout iteration:

1. **Policy forward passes** (multiple steps): generate a trajectory. Stateful (KV cache or denoising context). Might need to persist logprobs at each step (for importance weighting in PPO). Might need to persist intermediate states (for trajectory-level credit assignment).
2. **Reward forward pass:** run the completed trajectory through the reward model (which might be the same base model with a different adapter — level 5). Produces a scalar or sequence of rewards. This is a separate inference call with its own state.
3. **Gradient computation:** Now you need the *backward pass*, which means you need activations from step 1 to be either checkpointed or recomputed. This brings in activation checkpointing state — you're selectively saving intermediate activations during the forward pass to avoid recomputing everything during backward.
4. **Weight update:** Optimizer state (Adam moments, learning rate schedule) is now part of the system's persistent state. Per-parameter, two additional tensors (m and v) that persist across iterations. For LoRA, this is manageable (small). For full fine-tuning, it's 2× the model size in additional state.
5. **Loop:** Go to 1, but now with updated weights (the live weight update problem from above).

And if you're doing this on a serving system that's simultaneously handling inference requests with the weights that are being updated... now your state lifecycle has: serving weights (current), rollout weights (might be same or snapshotted), optimizer state, trajectory buffers, reward computation buffers, gradient accumulation buffers — all coexisting with live inference KV caches and request queues.

**This is a genuine and substantial change to the system's state lifecycle.** It's not that the forward pass changed. It's that the forward pass is now embedded in a larger stateful loop that maintains optimizer state, trajectory history, checkpointed activations, and weight versioning, all of which have lifecycles that span multiple forward passes and interact with the inference serving path.

---

## The real punchline

The forward pass composition is remarkably invariant across all of these scenarios. The *system state lifecycle* is not. And the failure mode you're describing in your agents is almost certainly: they look at the forward pass, see it's "the same," and conclude nothing interesting is happening — missing that the operationally significant complexity lives in the *state management around* the forward pass: what's persisted, what's mutable, what's versioned, what has consistency requirements, what competes for memory, and what has lifecycle dependencies on other state.

Documenting the forward pass correctly is table stakes. Documenting the state lifecycle of the *system* that hosts the forward pass is where the actual engineering lives, and it's orthogonal enough to the forward pass that an agent fixated on "but the matmuls are the same" will miss it entirely.
=====
so, can we get this as a relatively compact rollup? the outer specifications which have huge effects on state lifecycle, said plainly, as a collection of 'abstract sounding' outer specifications which have absolute primacy over (and in fact override and must trigger rewriting and reimplementation of) 'function level' implementations between functionally equivalent forms?
=====
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
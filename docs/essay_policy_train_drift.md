# Essay: Policy vs Training Implementation Drift in futudiffu

The document "Policy vs Training Implementation Drift Analysis" (2026-02-17) is a post-hoc audit of the futudiffu codebase that identifies five divergences between two paths through the same model: the rollout path that generates trajectory data, and the training path that computes REINFORCE policy gradients and BTRM reward scores against that data. Of these five divergences, two are classified as correctness bugs that produce wrong gradients. The core problem is not any single implementation error but rather a systemic condition: two code paths that must agree on the semantics of a forward pass have drifted apart silently, because nothing in the architecture enforces that agreement. The rollout path performs classifier-free guidance (CFG) -- a B=2 forward with positive and negative conditioning, followed by a linear combination of the two outputs -- while the training path duplicates the positive conditioning into both batch slots and takes only the first output, performing no CFG combination at all. The training path therefore optimizes policy parameters to match a reference model in a regime the policy will never actually operate in during inference. This is the document's central finding, and it is classified as critical.

The architectural issues the document raises go beyond the CFG bug itself. The drift exists because the rollout path (in `sampling.py`) and the training path (in `training_utils.py`) were written as separate functions that each independently construct their conditioning, call the compiled model forward, and interpret the output. There is no shared abstraction that enforces "a forward pass with CFG looks like this." The conditioning setup in the rollout path calls `pad_and_batch_cond(pos_cond, neg_cond)`, while the training path calls `pad_and_batch_cond(conditioning, conditioning)` -- a duplication that looks intentional in isolation but is semantically wrong in context. The sigma-indexing bug (divergence #2) has a similar structural cause: the checkpoint callback fires after the Euler step, saving x_{t+1}, but the training code evaluates the model at sigma_t rather than sigma_{t+1}. This is not a subtle theoretical disagreement; it is a straightforward off-by-one that persisted because the callback position and the sigma index were decided in different files by different logical concerns. The document also notes that the BTRM reward model was trained exclusively on pos_cond hidden states, never seeing the neg_cond branch that constitutes half of every real forward pass. This is less a bug than an omission, but it illustrates the same pattern: each subsystem was built to be locally correct without a cross-cutting invariant that all subsystems must respect.

The most critical actionable finding is divergence #1, the CFG mismatch. Its fix requires threading `neg_cond` and the `cfg` scale through the entire RPC chain -- from `train.py` on the client side, through `client.py`'s RPC interface, into `server.py`'s dispatch, and finally into `training_utils.py`'s `compute_reinforce_step` and its helper `_prepare_packed_single_state`. This is a five-file change that touches the protocol boundary between client and server, which means it cannot be done as a local patch. The sigma-indexing fix (divergence #2) is, by contrast, a one-line change -- use `sigmas[step_idx + 1]` instead of `sigmas[step_idx]` -- but the document correctly notes that for sparse step sampling with large sigma gaps, the systematic overestimation of the denominator in the log-ratio is non-trivial. Together, these two fixes would eliminate the two sources of wrong gradients. The BTRM neg_cond inclusion (divergence #3) is framed as a quality improvement that doubles training data per trajectory at zero additional generation cost, which makes it a high-value low-cost change once the CFG plumbing is in place.

These issues are a case study in a failure mode endemic to mid-sized codebases undergoing incremental feature extension. The futudiffu codebase grew from a ComfyUI port into a custom-kernel inference server, then acquired a BTRM training pipeline, then a REINFORCE policy optimizer, each layer building on the last. At each stage, the new feature was tested against its own local correctness criteria -- does the forward pass run, does the loss decrease, does the gradient flow -- without a regression harness that asserts semantic equivalence between the rollout and training forward passes. The rollout path was the "original" code; the training path was the "new" code that reused the same model but constructed its inputs differently. In a small codebase, this divergence would be caught by inspection. In a large codebase, it would be caught by a dedicated integration test team or by formal interface contracts. In a mid-sized codebase -- large enough that no single person holds every function signature in working memory, small enough that formal contracts feel like overhead -- it falls through. The document is, in effect, the output of the integration test that should have existed before the first training run.

The document implies a clear set of next steps: fix the CFG bug first (it invalidates all policy gradient computation), fix the sigma index second (one line, immediate), then extend BTRM training to include neg_cond hidden states. But the deeper implication is structural. The five critical files listed at the end of the document -- `training_utils.py`, `sampling.py`, `train.py`, `client.py`, `server.py` -- form the spine of the system, and the document demonstrates that their implicit contracts with each other are not enforced by any mechanism other than the developer's memory. A forward pass that means different things in different contexts is not a stable foundation for iterative policy optimization. The next priority, beyond the specific fixes, is likely the extraction of a shared "conditioned forward with optional CFG" primitive that both paths call, eliminating the possibility of drift by construction rather than by audit.

---

## Appendix: Supporting Quotations from Source Document

**On the severity classification of the five divergences:**

> Five divergences between the rollout (data generation) path and the training (REINFORCE + BTRM) path. Two are correctness bugs that produce wrong gradients.

**On the rollout path's proper CFG construction:**

> `refined_caps` built from `pad_and_batch_cond(pos_cond, neg_cond)` -- proper CFG conditioning. The euler step uses CFG-combined denoised prediction. The saved checkpoint x_{t+1} is shaped by CFG.

**On the training path's duplication of pos_cond (the critical bug):**

> ```
> pad_and_batch_cond(conditioning, conditioning)  # pos_cond DUPLICATED
> # Both batch elements see identical positive conditioning
> ```

**On the training path's omission of CFG combination:**

> No CFG combination: raw model(x_t, sigma, pos_cond) output. Client (`train.py` L601-607) sends only `pos_cond[:1]`, never neg_cond.

**On the consequence for REINFORCE gradients:**

> Both `pi_denoised` and `ref_denoised` are computed WITHOUT CFG, on a latent that was generated WITH CFG. The gradient optimizes the policy to match the reference in the pos_cond-only regime, not in the CFG regime that actually produced the trajectories and rewards.

**On the sigma-indexing off-by-one:**

> `compute_reinforce_step` evaluates `model(checkpoint[step_i], sigmas[step_i])`, which is `model(x_{t+1}, sigma_t)`. But x_{t+1} corresponds to noise level sigma_{t+1}, not sigma_t.

**On the systematic bias introduced by the sigma mismatch:**

> sigma_t > sigma_{t+1}, so the denominator `2*sigma_t^2` in the log-ratio is systematically too large, underweighting the gradient signal. For sparse steps with large sigma gaps, this is non-trivial.

**On the BTRM neg_cond omission as a missed opportunity:**

> The empty-string conditioning (neg_cond) is a real in-distribution input to the model. The neg_cond hidden states carry quality-discriminative information (unconditional prediction quality also degrades with attention quantization and step count).

**On the free data doubling from including neg_cond:**

> Include neg_cond as separate training examples with the SAME quality labels. This doubles training data per trajectory at zero additional generation cost.

**On the scope of the CFG fix across the RPC boundary:**

> Critical Files: `src/futudiffu/training_utils.py` -- compute_reinforce_step, _prepare_packed_single_state; `src/futudiffu/sampling.py` -- sample_euler_packed callback position; `scripts/train.py` -- client-side: pass neg_cond + cfg to accumulate_policy_gradients; `src/futudiffu/client.py` -- RPC interface: add neg_cond + cfg params; `src/futudiffu/server.py` -- RPC dispatch: thread new params through.

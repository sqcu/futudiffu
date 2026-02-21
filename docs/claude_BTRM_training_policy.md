### Both reward models and policy models are fine-tuned *models*

A BTRM reward model is: backbone + LoRA adapter (r_theta) + score unembedder.
A policy model is: backbone + LoRA adapter (p_theta).

Both are MODELS with trainable parameters INSIDE the backbone (via LoRA).
They are NOT probes. They are NOT linear classifiers on frozen features.

Training either model requires:
- Full forward pass through the backbone with gradients enabled
- The computation graph must flow through the adapter's LoRA matrices
- Backpropagation must reach the adapter parameters
- The optimizer must include adapter parameters

### What is NOT acceptable

The low-rank PEFT adapter training implementation does NOT give us latitude to:
- Extract hidden states and train a head on detached features
- Use a linear probe instead of a trained model
- Run the backbone under no_grad/inference_mode during training
- Create an optimizer with only head/unembedder parameters
- Delete the backbone before training ("extract then free" pattern)

These patterns are correct for probe training but WRONG for adapter training.
The recurring defect (6+ occurrences) of treating the adapter as optional
comes from the probe mental model. It is structurally prevented by using
BTRMCompoundModel, which enforces the coupling.

### Naming convention

- "BTRM model" = the compound triple (backbone + adapter + unembedder)
- "Score unembedder" = the output layer (RMSNorm + Linear + tanh_cap)
- "BTRMHead" is deprecated terminology -- use "score unembedder" or `ScoreUnembedder`
- Never instantiate a score unembedder without a compound model in training code
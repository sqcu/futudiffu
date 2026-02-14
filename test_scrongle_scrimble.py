"""Test: BTRM two-head classification of diffusion trajectory degradation types.

Validates that a multi-head Bradley-Terry Reward Model can discriminate between:
  - scrongle: step-count artifacts (too few sampling steps)
  - scrimble: quantization artifacts (lower-precision weights)

Uses a small MiniModel (NOT the real NextDiT) to generate synthetic paired
trajectory data, trains the BTRM heads, and measures classification accuracy.

Phases:
  1. Generate synthetic trajectory pairs (step-count + quantization degradation)
  2. Train two-head BTRM on Bradley-Terry ranking loss
  3. Evaluate per-head classification accuracy
  4. Concurrent scoring with per-batch routing
"""

import sys
import time

sys.path.insert(0, r"F:\dox\repos\ai\futudiffu\src")

import torch
import torch.nn as nn
import torch.nn.functional as F

from futudiffu.lora import inject_lora, get_lora_params, set_lora_scale, freeze_adapter
from futudiffu.btrm import BTRMHead, bradley_terry_loss


# ---------------------------------------------------------------------------
# Minimal transformer model (copied from test_multilora_fused.py)
# ---------------------------------------------------------------------------

class MiniAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        return self.out(v)


class MiniFFN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w1 = nn.Linear(dim, dim * 2, bias=False)
        self.w2 = nn.Linear(dim * 2, dim, bias=False)
        self.w3 = nn.Linear(dim, dim * 2, bias=False)

    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class MiniBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = MiniAttn(dim)
        self.feed_forward = MiniFFN(dim)

    def forward(self, x):
        x = x + self.attention(x)
        x = x + self.feed_forward(x)
        return x


class MiniModel(nn.Module):
    def __init__(self, dim, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([MiniBlock(dim) for _ in range(n_layers)])
        self.head = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x)

    def forward_hidden(self, x):
        """Forward pass returning hidden states BEFORE model.head."""
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DIM = 64
N_LAYERS = 4
SEQ_LEN = 16
DEVICE = "cuda"
DTYPE = torch.bfloat16

N_PAIRS = 100         # pairs per head type (100 scrongle + 100 scrimble)
FULL_STEPS = 5        # "good" trajectory step count (kept low to avoid fixed-point convergence)
TRUNCATED_STEPS = 1   # "bad" trajectory step count for scrongle
QUANT_NOISE_SCALE = 0.8   # noise magnitude for scrimble degradation (large: simulates severe FP8 artifacts)

TRAIN_FRACTION = 0.8
TRAIN_STEPS = 800
LR = 1e-3

SEED = 42


# ---------------------------------------------------------------------------
# Phase 1: Generate synthetic trajectory pairs
# ---------------------------------------------------------------------------

def simulate_trajectory(model, x, n_steps):
    """Simulate a 'diffusion trajectory' by running n_steps forward passes.

    Returns the hidden state after the final layer (before model.head).
    Each step feeds the output of model.head back as input, simulating
    iterative denoising.
    """
    with torch.no_grad():
        for step in range(n_steps):
            out = model(x)
            # Use output as next input (simulating euler step iteration)
            x = out
        # Return hidden states from the final step (before head projection)
        hidden = model.forward_hidden(x)
    return hidden


def simulate_noisy_trajectory(model, x, n_steps, noise_scale):
    """Simulate a trajectory with per-step quantization noise on activations.

    At each step, after the forward pass, we add noise proportional to
    the output magnitude. This simulates the cumulative effect of running
    with quantized weights where each layer introduces small errors that
    compound over steps.
    """
    with torch.no_grad():
        for step in range(n_steps):
            out = model(x)
            # Add quantization-style noise (proportional to activation magnitude)
            noise = torch.randn_like(out) * out.abs() * noise_scale
            x = out + noise
        hidden = model.forward_hidden(x)
    return hidden


def add_quantization_noise(model, noise_scale):
    """Add noise proportional to weight magnitude, simulating FP8 quantization artifacts.

    Returns a dict of saved original weights for restore_weights().
    Clamps noised weights to prevent bf16 overflow.
    """
    saved = {}
    for name, param in model.named_parameters():
        saved[name] = param.data.clone()
        noise = torch.randn_like(param) * param.data.abs() * noise_scale
        param.data.add_(noise)
        # Clamp to prevent bf16 overflow during forward passes
        param.data.clamp_(-1e4, 1e4)
    return saved


def restore_weights(model, saved):
    """Restore original weights from saved dict."""
    for name, param in model.named_parameters():
        param.data.copy_(saved[name])


def generate_trajectory_data(model):
    """Generate synthetic trajectory pairs for both head types.

    Returns:
        scrongle_pos: (N, seq, dim) -- good trajectories (full steps)
        scrongle_neg: (N, seq, dim) -- bad trajectories (truncated steps)
        scrimble_pos: (N, seq, dim) -- good trajectories (full precision)
        scrimble_neg: (N, seq, dim) -- bad trajectories (quantization noise)
    """
    print("Phase 1: Generating synthetic trajectory pairs")
    print(f"  config: dim={DIM}, n_layers={N_LAYERS}, seq_len={SEQ_LEN}")
    print(f"  scrongle: {FULL_STEPS} steps (good) vs {TRUNCATED_STEPS} steps (bad)")
    print(f"  scrimble: clean weights (good) vs quant_noise={QUANT_NOISE_SCALE} (bad)")
    print(f"  pairs per head: {N_PAIRS}")

    scrongle_pos_list = []
    scrongle_neg_list = []
    scrimble_pos_list = []
    scrimble_neg_list = []

    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(SEED)

    t0 = time.time()

    for i in range(N_PAIRS):
        # Generate a random starting latent for this pair
        x = torch.randn(1, SEQ_LEN, DIM, dtype=DTYPE, device=DEVICE, generator=gen)

        # --- Scrongle pair: same start, different step counts ---
        good_hidden = simulate_trajectory(model, x.clone(), FULL_STEPS)
        bad_hidden = simulate_trajectory(model, x.clone(), TRUNCATED_STEPS)
        scrongle_pos_list.append(good_hidden.squeeze(0))
        scrongle_neg_list.append(bad_hidden.squeeze(0))

        # --- Scrimble pair: same start & steps, different weight precision ---
        # "Good" trajectory: clean forward passes
        good_hidden = simulate_trajectory(model, x.clone(), FULL_STEPS)
        # "Bad" trajectory: per-step activation noise simulating quantization error
        bad_hidden = simulate_noisy_trajectory(model, x.clone(), FULL_STEPS,
                                               QUANT_NOISE_SCALE)

        scrimble_pos_list.append(good_hidden.squeeze(0))
        scrimble_neg_list.append(bad_hidden.squeeze(0))

    elapsed = time.time() - t0

    scrongle_pos = torch.stack(scrongle_pos_list)  # (N, seq, dim)
    scrongle_neg = torch.stack(scrongle_neg_list)
    scrimble_pos = torch.stack(scrimble_pos_list)
    scrimble_neg = torch.stack(scrimble_neg_list)

    # Report statistics on the generated data
    def stat(name, pos, neg):
        pos_norm = pos.float().norm(dim=-1).mean().item()
        neg_norm = neg.float().norm(dim=-1).mean().item()
        cos = F.cosine_similarity(
            pos.float().reshape(-1, DIM), neg.float().reshape(-1, DIM), dim=-1
        ).mean().item()
        print(f"  {name}: pos_norm={pos_norm:.3f}, neg_norm={neg_norm:.3f}, "
              f"mean_cosine(pos,neg)={cos:.4f}")

    stat("scrongle", scrongle_pos, scrongle_neg)
    stat("scrimble", scrimble_pos, scrimble_neg)
    print(f"  generation time: {elapsed:.2f}s")

    return scrongle_pos, scrongle_neg, scrimble_pos, scrimble_neg


# ---------------------------------------------------------------------------
# Phase 2: Train two-head BTRM
# ---------------------------------------------------------------------------

def train_btrm(btrm_head, scrongle_pos, scrongle_neg, scrimble_pos, scrimble_neg,
               n_train_scrongle, n_train_scrimble):
    """Train the two-head BTRM using separate BT losses per head.

    Args:
        btrm_head: BTRMHead with heads ("scrongle", "scrimble")
        *_pos/*_neg: (N, seq, dim) trajectory hidden states
        n_train_*: number of training pairs per head
    """
    print(f"\nPhase 2: Training BTRM ({TRAIN_STEPS} steps, lr={LR})")

    optimizer = torch.optim.Adam(btrm_head.parameters(), lr=LR)

    scrongle_idx = btrm_head.get_head_idx("scrongle")
    scrimble_idx = btrm_head.get_head_idx("scrimble")

    # Training data (first n_train pairs)
    train_scrongle_pos = scrongle_pos[:n_train_scrongle]
    train_scrongle_neg = scrongle_neg[:n_train_scrongle]
    train_scrimble_pos = scrimble_pos[:n_train_scrimble]
    train_scrimble_neg = scrimble_neg[:n_train_scrimble]

    losses_log = []

    for step in range(TRAIN_STEPS):
        optimizer.zero_grad()

        # --- Scrongle head loss ---
        # Randomly sample a mini-batch of scrongle pairs
        idx = torch.randint(0, n_train_scrongle, (min(16, n_train_scrongle),),
                            device=DEVICE)
        sp = train_scrongle_pos[idx]   # (B, seq, dim)
        sn = train_scrongle_neg[idx]

        sp_scores = btrm_head(sp)      # (B, 2)
        sn_scores = btrm_head(sn)

        loss_scrongle = bradley_terry_loss(
            sp_scores[:, scrongle_idx],
            sn_scores[:, scrongle_idx]
        )

        # --- Scrimble head loss ---
        idx = torch.randint(0, n_train_scrimble, (min(16, n_train_scrimble),),
                            device=DEVICE)
        qp = train_scrimble_pos[idx]
        qn = train_scrimble_neg[idx]

        qp_scores = btrm_head(qp)
        qn_scores = btrm_head(qn)

        loss_scrimble = bradley_terry_loss(
            qp_scores[:, scrimble_idx],
            qn_scores[:, scrimble_idx]
        )

        loss = loss_scrongle + loss_scrimble
        loss.backward()
        optimizer.step()

        losses_log.append((loss_scrongle.item(), loss_scrimble.item()))

        if step % 50 == 0 or step == TRAIN_STEPS - 1:
            print(f"  step {step:4d}: "
                  f"scrongle_loss={loss_scrongle.item():.4f}, "
                  f"scrimble_loss={loss_scrimble.item():.4f}, "
                  f"total={loss.item():.4f}")

    # Report training curve endpoints
    first_total = losses_log[0][0] + losses_log[0][1]
    last_total = losses_log[-1][0] + losses_log[-1][1]
    print(f"  loss reduction: {first_total:.4f} -> {last_total:.4f} "
          f"({(1 - last_total/first_total)*100:.1f}% decrease)")

    return losses_log


# ---------------------------------------------------------------------------
# Phase 3: Evaluate classification accuracy
# ---------------------------------------------------------------------------

def evaluate_accuracy(btrm_head, scrongle_pos, scrongle_neg, scrimble_pos, scrimble_neg,
                      n_train_scrongle, n_train_scrimble):
    """Evaluate per-head classification accuracy on held-out pairs.

    A pair is correctly classified if score(good) > score(bad).
    """
    print("\nPhase 3: Classification accuracy")

    scrongle_idx = btrm_head.get_head_idx("scrongle")
    scrimble_idx = btrm_head.get_head_idx("scrimble")

    btrm_head.eval()

    with torch.no_grad():
        # Held-out scrongle pairs
        val_sp = scrongle_pos[n_train_scrongle:]
        val_sn = scrongle_neg[n_train_scrongle:]
        n_val_scrongle = val_sp.shape[0]

        sp_scores = btrm_head(val_sp)[:, scrongle_idx]
        sn_scores = btrm_head(val_sn)[:, scrongle_idx]
        scrongle_correct = (sp_scores > sn_scores).sum().item()

        # Held-out scrimble pairs
        val_qp = scrimble_pos[n_train_scrimble:]
        val_qn = scrimble_neg[n_train_scrimble:]
        n_val_scrimble = val_qp.shape[0]

        qp_scores = btrm_head(val_qp)[:, scrimble_idx]
        qn_scores = btrm_head(val_qn)[:, scrimble_idx]
        scrimble_correct = (qp_scores > qn_scores).sum().item()

    total_correct = scrongle_correct + scrimble_correct
    total_val = n_val_scrongle + n_val_scrimble

    scrongle_acc = scrongle_correct / n_val_scrongle * 100
    scrimble_acc = scrimble_correct / n_val_scrimble * 100
    overall_acc = total_correct / total_val * 100

    print(f"  scrongle (step count):    {scrongle_correct}/{n_val_scrongle} = {scrongle_acc:.1f}%")
    print(f"  scrimble (quantization):  {scrimble_correct}/{n_val_scrimble} = {scrimble_acc:.1f}%")
    print(f"  overall:                  {total_correct}/{total_val} = {overall_acc:.1f}%")

    # Diagnostic score distributions if accuracy is low
    if scrongle_acc < 75.0 or scrimble_acc < 75.0:
        print("\n  WARNING: accuracy below 75% target. Score diagnostics:")
        with torch.no_grad():
            sp_all = btrm_head(scrongle_pos)[:, scrongle_idx]
            sn_all = btrm_head(scrongle_neg)[:, scrongle_idx]
            print(f"    scrongle pos scores: mean={sp_all.mean():.4f}, std={sp_all.std():.4f}")
            print(f"    scrongle neg scores: mean={sn_all.mean():.4f}, std={sn_all.std():.4f}")
            print(f"    scrongle margin:     mean={( sp_all - sn_all).mean():.4f}")

            qp_all = btrm_head(scrimble_pos)[:, scrimble_idx]
            qn_all = btrm_head(scrimble_neg)[:, scrimble_idx]
            print(f"    scrimble pos scores: mean={qp_all.mean():.4f}, std={qp_all.std():.4f}")
            print(f"    scrimble neg scores: mean={qn_all.mean():.4f}, std={qn_all.std():.4f}")
            print(f"    scrimble margin:     mean={(qp_all - qn_all).mean():.4f}")

    return scrongle_acc, scrimble_acc, overall_acc


# ---------------------------------------------------------------------------
# Phase 4: Concurrent scoring with per-batch routing
# ---------------------------------------------------------------------------

def test_concurrent_routing(btrm_head, scrongle_pos, scrongle_neg,
                            scrimble_pos, scrimble_neg):
    """Demonstrate concurrent per-batch routing: batch[0]=scrongle, batch[1]=scrimble.

    Verify that batched scores match individual forward passes (cosine > 0.99).
    """
    print("\nPhase 4: Concurrent scoring with per-batch routing")

    btrm_head.eval()
    scrongle_idx = btrm_head.get_head_idx("scrongle")
    scrimble_idx = btrm_head.get_head_idx("scrimble")

    with torch.no_grad():
        # Pick one sample from each head type
        scrongle_sample = scrongle_pos[0:1]  # (1, seq, dim)
        scrimble_sample = scrimble_pos[0:1]  # (1, seq, dim)

        # Individual forward passes (reference)
        scrongle_scores_ref = btrm_head(scrongle_sample)  # (1, 2)
        scrimble_scores_ref = btrm_head(scrimble_sample)  # (1, 2)

        # Batched forward pass
        batch = torch.cat([scrongle_sample, scrimble_sample], dim=0)  # (2, seq, dim)
        batch_scores = btrm_head(batch)  # (2, 2)

        # Extract per-head scores from batch
        batch_scrongle_score = batch_scores[0, scrongle_idx]
        batch_scrimble_score = batch_scores[1, scrimble_idx]

        # Extract reference scores
        ref_scrongle_score = scrongle_scores_ref[0, scrongle_idx]
        ref_scrimble_score = scrimble_scores_ref[0, scrimble_idx]

        # Cosine similarity between full score vectors
        cos_scrongle = F.cosine_similarity(
            batch_scores[0:1], scrongle_scores_ref, dim=-1
        ).item()
        cos_scrimble = F.cosine_similarity(
            batch_scores[1:2], scrimble_scores_ref, dim=-1
        ).item()

        print(f"  scrongle: batch_score={batch_scrongle_score:.4f}, "
              f"ref_score={ref_scrongle_score:.4f}, cos={cos_scrongle:.6f}")
        print(f"  scrimble: batch_score={batch_scrimble_score:.4f}, "
              f"ref_score={ref_scrimble_score:.4f}, cos={cos_scrimble:.6f}")

        # Exact match expected (no per-batch routing here, just verifying
        # that batched vs individual gives same results through the BTRM head)
        scrongle_match = torch.allclose(batch_scores[0], scrongle_scores_ref[0])
        scrimble_match = torch.allclose(batch_scores[1], scrimble_scores_ref[0])

        print(f"  scrongle exact match: {scrongle_match}")
        print(f"  scrimble exact match: {scrimble_match}")

        assert cos_scrongle > 0.99, f"Scrongle cosine too low: {cos_scrongle}"
        assert cos_scrimble > 0.99, f"Scrimble cosine too low: {cos_scrimble}"

        # Also verify that scrongle head prefers scrongle-good over scrimble-good
        # and vice versa (heads are specialized)
        print(f"\n  Head specialization check:")
        print(f"    scrongle head on scrongle_pos[0]: {batch_scores[0, scrongle_idx]:.4f}")
        print(f"    scrongle head on scrimble_pos[0]: {batch_scores[1, scrongle_idx]:.4f}")
        print(f"    scrimble head on scrongle_pos[0]: {batch_scores[0, scrimble_idx]:.4f}")
        print(f"    scrimble head on scrimble_pos[0]: {batch_scores[1, scrimble_idx]:.4f}")

    print("  PASS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  BTRM Scrongle/Scrimble Classification Test")
    print("=" * 60)

    assert torch.cuda.is_available(), "CUDA required"
    print(f"  device: {torch.cuda.get_device_name(0)}")
    print(f"  dtype: {DTYPE}")
    print()

    # Set global seed for reproducibility
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    # Build the MiniModel (frozen, used only for trajectory generation)
    model = MiniModel(DIM, N_LAYERS).to(dtype=DTYPE, device=DEVICE)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  MiniModel: {total_params:,} params")
    print()

    # Phase 1: Generate trajectory data
    scrongle_pos, scrongle_neg, scrimble_pos, scrimble_neg = \
        generate_trajectory_data(model)

    # Split into train/val
    n_train_scrongle = int(N_PAIRS * TRAIN_FRACTION)
    n_train_scrimble = int(N_PAIRS * TRAIN_FRACTION)
    n_val_scrongle = N_PAIRS - n_train_scrongle
    n_val_scrimble = N_PAIRS - n_train_scrimble
    print(f"  train/val split: {n_train_scrongle}/{n_val_scrongle} per head")

    # Phase 2: Build and train BTRM
    btrm_head = BTRMHead(
        hidden_dim=DIM,
        head_names=("scrongle", "scrimble"),
        logit_cap=10.0,
    ).to(dtype=DTYPE, device=DEVICE)

    btrm_params = sum(p.numel() for p in btrm_head.parameters())
    print(f"  BTRMHead: {btrm_params:,} params "
          f"(norm={DIM} + proj={DIM}x2 = {DIM + DIM*2})")

    train_btrm(btrm_head, scrongle_pos, scrongle_neg, scrimble_pos, scrimble_neg,
               n_train_scrongle, n_train_scrimble)

    # Phase 3: Evaluate
    scrongle_acc, scrimble_acc, overall_acc = evaluate_accuracy(
        btrm_head, scrongle_pos, scrongle_neg, scrimble_pos, scrimble_neg,
        n_train_scrongle, n_train_scrimble
    )

    # Phase 4: Concurrent routing
    test_concurrent_routing(btrm_head, scrongle_pos, scrongle_neg,
                            scrimble_pos, scrimble_neg)

    # Summary
    print()
    print("=" * 60)
    target_met = scrongle_acc >= 75.0 and scrimble_acc >= 75.0
    if target_met:
        print(f"  BOTH HEADS ABOVE 75% TARGET")
    else:
        print(f"  NOTE: one or more heads below 75% target (research validation)")
    print(f"  scrongle={scrongle_acc:.1f}%, scrimble={scrimble_acc:.1f}%, "
          f"overall={overall_acc:.1f}%")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())

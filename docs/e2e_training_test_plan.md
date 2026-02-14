# E2E Training Test Plan: Real Trajectory Data

Testing the server-based training pipeline (`smoke_test_e2e_training.py` architecture)
against the 50 trajectories already generated in `btrm_dataset/`.

---

## 1. Data Inventory

### 50 trajectories in `btrm_dataset/latents/`

| Batch | Type | Attn Backend | Step Count | Count | Traj IDs     |
|-------|------|-------------|------------|-------|-------------- |
| 0     | t2i  | SDPA        | 30         | 10    | 000000-000009 |
| 1     | t2i  | Sage        | 30         | 10    | 000010-000019 |
| 2     | t2i  | SDPA        | 10-22      | 10    | 000020-000029 |
| 3     | t2i  | Sage        | 8-21       | 10    | 000030-000039 |
| 0     | i2i  | SDPA        | 30         | 5     | 000040-000044 |
| 1     | i2i  | Sage        | 30         | 5     | 000045-000049 |

### Per-trajectory files

Each `traj_NNNNNN/` contains:
- `meta.json` — seed, prompt, n_steps, precision, etc.
- `step_00.pt`, `step_04.pt`, `step_09.pt`, `step_14.pt`, `step_19.pt`, `step_24.pt`, `step_29.pt` — intermediate latents
- `final.pt` — final denoised latent
- Each `.pt` is a `(1, 16, 104, 160)` BF16 tensor (~522KB)

### Prompt distribution

13 unique prompt_idx values across the 40 t2i trajectories. Some prompts appear
across multiple batches (enabling cross-backend pairing by prompt):

| prompt_idx | Prompt (short) | Batches |
|-----------|----------------|---------|
| 0  | laser shark sega saturn    | B0, B3 |
| 1  | laser shark breaching      | B2, B3 |
| 2  | laser shark cyberpunk      | B1, B3 |
| 4  | tiny laser shark fishbowl  | B2, B3 |
| 5  | laser shark chrome         | B0, B1 |
| 6  | neon sign OPEN 24 HOURS    | B0, B1 |
| 7  | handwritten letter         | B0 (x3)|
| 8  | chalkboard E=mc^2          | B1, B2 |
| 10 | cat on books               | B0     |
| 11 | astronaut horse            | B1     |
| 14 | pocket watch               | B2, B3 |
| 18 | oil painting mountain lake | B0, B1, B2 |
| 19 | vaporwave Greek ruins      | B1     |
| 20 | Bauhaus geometric          | B0, B1, B2, B3 |
| 21 | wolf double exposure       | B1, B2 |
| 23 | isometric pixel art        | B0, B3 |

---

## 2. Pairing Strategy

### Scrimble pairs (attention quality — head_idx=0 "bit_quality")

Scrimble measures: SDPA (gold) vs Sage (quantized attention). The BTRM should
score SDPA higher.

**Constraint**: Our dataset has no seed-paired SDPA/Sage trajectories. Seeds differ
across batches. We pair by prompt_idx instead, accepting that seed variation adds
noise to the supervision signal. This is fine for a smoke test — the BTRM head just
needs to learn *some* systematic difference between SDPA and Sage outputs.

Viable prompt-based pairs (B0 SDPA 30-step vs B1 Sage 30-step):

| SDPA traj | Sage traj | prompt_idx | Prompt |
|-----------|-----------|-----------|--------|
| 000000    | 000014    | 20 | Bauhaus geometric |
| 000003    | 000015    | 18 | oil painting mountain lake |
| 000007    | 000018    | 5  | laser shark chrome |
| 000009    | 000019    | 6  | neon sign OPEN 24 HOURS |

4 scrimble pairs from t2i. Could also mine i2i pairs if same image_file
appears in both SDPA and Sage batches (e.g., `1bit redraw.png` in traj 041/042
vs 046, `00500-...nightmode2.png` in 040/044 vs 047), though prompts differ.

### Scrongle pairs (step quality — head_idx=1 "step_quality")

Scrongle measures: 30-step (gold) vs reduced-step (degraded). The BTRM should
score 30-step higher.

Pair by prompt_idx, matching B0/B1 (30-step) against B2/B3 (varied-step):

| 30-step traj | Low-step traj | Steps | prompt_idx | Prompt |
|-------------|--------------|-------|-----------|--------|
| 000000 (SDPA) | 000021 (SDPA) | 17 | 20 | Bauhaus geometric |
| 000000 (SDPA) | 000023 (SDPA) | 22 | 20 | Bauhaus geometric |
| 000003 (SDPA) | 000026 (SDPA) | 22 | 18 | oil painting |
| 000003 (SDPA) | 000029 (SDPA) | 22 | 18 | oil painting |
| 000014 (Sage) | 000032 (Sage) | 10 | 20 | Bauhaus geometric |
| 000014 (Sage) | 000036 (Sage) | 20 | 20 | Bauhaus geometric |
| 000014 (Sage) | 000039 (Sage) | 21 | 20 | Bauhaus geometric |
| 000015 (Sage) | [none] | — | 18 | oil painting |
| 000010 (Sage) | 000033 (Sage) | 17 | 2 | laser shark cyberpunk |

~8 scrongle pairs.

### Summary: ~12 usable pairs total (4 scrimble + 8 scrongle)

This is small but sufficient for a smoke test. The goal is not to train a good
BTRM — it's to verify the plumbing works end-to-end.

---

## 3. Test Architecture

### Offline BTRM training (no live rollouts)

Unlike `smoke_test_e2e_training.py` which generates rollouts on-the-fly via the
inference server, this test loads pre-computed trajectory latents from disk and
sends them to the server's `train_btrm_step` endpoint. The server runs
backbone forward + BTRM scoring + BT loss + backward + optimizer step atomically.

```
                     btrm_dataset/latents/
                           |
                    load .pt files
                           |
     Client                v               Server
  +-----------+    train_btrm_step()   +------------+
  | scheduling|-- labeled examples --> | FP8 NextDiT|
  | (no GPU)  |<-- scalar metrics ---  | + BTRMHead |
  +-----------+                        | + optimizer|
                                       +------------+
```

### Full e2e policy test (live rollouts + weight push)

After BTRM training, test the policy optimization loop:

```
  Client                                        Server
+-----------+    sample_trajectory()        +------------+
| scheduling|<-- trajectories -----------  | FP8 NextDiT|
| (no GPU)  |                              | + LoRA     |
+-----------+                              | + BTRMHead |
     |                                     | + optimzrs |
  score_btrm() ---------------------------->           |
     |<-- scalar scores -----------------------        |
  advantages                                           |
  accumulate_policy_gradients() x K ------->  [grads]  |
  policy_optimizer_step() ----------------->  [step]   |
     |                                                 |
     v [repeat]                                        v
```

---

## 4. Implementation: Trajectory Loader

New file: `test_e2e_real_trajectories.py`

### Data loading

```python
def load_trajectory(traj_dir: str) -> dict[str, torch.Tensor]:
    """Load all .pt files from a trajectory directory."""
    result = {}
    for pt_file in Path(traj_dir).glob("*.pt"):
        key = pt_file.stem  # "step_00", "step_04", ..., "final"
        result[key] = torch.load(pt_file, weights_only=True)
    return result

def load_manifest(dataset_dir: str) -> list[dict]:
    """Load manifest.json and return records list."""
    with open(Path(dataset_dir) / "manifest.json") as f:
        return json.load(f)["records"]
```

### Pair construction

```python
def build_scrimble_pairs(records: list[dict]) -> list[tuple[dict, dict]]:
    """Build SDPA(+) vs Sage(-) pairs matched by prompt_idx, both 30-step."""
    by_prompt: dict[int, dict[str, list[dict]]] = {}
    for r in records:
        if r["type"] == "t2i" and r["n_steps"] == 30:
            pid = r["prompt_idx"]
            by_prompt.setdefault(pid, {"sdpa": [], "sage": []})
            by_prompt[pid][r["precision"]].append(r)
    pairs = []
    for pid, groups in by_prompt.items():
        for pos in groups["sdpa"]:
            for neg in groups["sage"]:
                pairs.append((pos, neg))
    return pairs

def build_scrongle_pairs(records: list[dict]) -> list[tuple[dict, dict]]:
    """Build 30-step(+) vs reduced-step(-) pairs matched by prompt_idx."""
    by_prompt: dict[int, dict[str, list[dict]]] = {}
    for r in records:
        if r["type"] == "t2i":
            pid = r["prompt_idx"]
            key = "full" if r["n_steps"] == 30 else "reduced"
            by_prompt.setdefault(pid, {"full": [], "reduced": []})
            by_prompt[pid][key].append(r)
    pairs = []
    for pid, groups in by_prompt.items():
        for pos in groups["full"]:
            for neg in groups["reduced"]:
                pairs.append((pos, neg))
    return pairs
```

---

## 5. BTRM Training Protocol (Phase 1)

### Setup

1. Start server: `python -m futudiffu.server --port 5555 --fp8-diff <path> --te <path> --vae <path>`
2. Client connects, encodes prompts (may not be needed if we use pre-cached states)
3. Inject `rtheta` LoRA on layers 28-29
4. Warmup compiled model
5. Create BTRM head on server via `inject_btrm_head(head_names, lr=...)`

### Training loop (10 steps)

For each macrobatch:
1. Sample examples from the pool
2. Load step checkpoints from each trajectory (e.g., `step_14.pt`)
3. Compute sigma for each step (from the trajectory's n_steps schedule)
4. Label each example with head_idx and is_positive
5. Send macrobatch to server via `train_btrm_step(examples)`
6. Server runs backbone + BTRM head + BT loss + backward + step atomically
7. Client receives scalar metrics: loss, per_head_accuracy

### Success criteria

- No NaN in losses
- BT loss decreases over 10 steps (or at least doesn't diverge)
- Both heads receive training signal (active_heads == 2 in some batches)
- Margins (pos_score - neg_score) trend positive

### Sigma computation for stored checkpoints

Each trajectory was generated with a specific n_steps. To construct labeled examples
for `train_btrm_step`, we need the sigma at each step:

```python
sigma_table = build_sigmas(shift=1.0, multiplier=1000.0)
sigmas = simple_scheduler(sigma_table, n_steps)
sigma_at_step_k = sigmas[k]
```

The step indices in our dataset are [0, 4, 9, 14, 19, 24, 29], so we can compute
sigma for any of those. Different n_steps trajectories have different sigma schedules.

---

## 6. Policy Optimization Protocol (Phase 2)

### Setup (after BTRM phase)

1. Freeze rtheta LoRA, set scale=0
2. Inject `ptheta` LoRA on all layers (init_b_std=0.01)
3. Warmup

### Training loop (10 iterations)

For each iteration:
1. Generate K=2 rollouts via `sample_trajectory()` (live, not from disk)
   - Use the laser shark prompt
   - Different seeds per rollout
   - n_steps=10 (fast, since this is a smoke test)
   - Save sparse step indices [0, 4, 9]
2. Score each rollout with server-side BTRM via `score_btrm()`
3. Compute group advantages
4. For each rollout: `accumulate_policy_gradients()` on server (gradients stay server-side)
5. `policy_optimizer_step()` on server (clip, step, zero)
6. Next iteration uses updated policy (server stepped optimizer in-place)

### Success criteria

- No NaN in rewards or gradients
- Gradient norm is finite and non-zero
- Weight push succeeds (no crash)
- Rewards may or may not increase (10 steps with K=2 is stochastic — trend
  not guaranteed). Primary goal: plumbing works, numbers are sane.

### Verification that weight push actually affects rollouts

At iterations 0 and 9, save the `final` latent. Compute cosine similarity.
If the weights are actually being applied, cos(final_0, final_9) < 1.0.
Even with the same seed, updated LoRA weights should change the output.

---

## 7. Data Gap Analysis

### What we have

- 50 trajectories with 7 intermediate checkpoints each
- Diverse prompts (13 unique prompt_idx)
- SDPA vs Sage backend variation
- Step count variation (8-30)
- i2i trajectories with varied denoise strengths

### What's missing for robust BTRM training

1. **No seed-paired SDPA/Sage**: Each batch uses different RNG seeds, so we
   can't isolate attention backend as the sole variable. Pairs are matched by
   prompt_idx but differ in seed. For a smoke test this is fine; for real BTRM
   training, we'd need same-seed same-prompt trajectories differing only in
   attention backend.

2. **No same-prompt same-backend step-count pairs**: Scrongle pairs are matched
   by prompt_idx but not by seed. Again fine for smoke testing.

3. **Limited i2i pairing**: The 10 i2i trajectories use different source images
   and prompts across SDPA/Sage batches. Only `00500-...nightmode2.png` and
   `1bit redraw.png` appear in both, but with different prompts.

4. **No explicit conditioning tensors stored**: The manifest stores the prompt
   text, not the encoded conditioning. We need to re-encode via the server's
   `encode_prompt()` endpoint each time (acceptable cost, one TE forward per
   unique prompt).

### Mitigation for the smoke test

Accept prompt-matched (not seed-matched) pairs. The signal is noisier but
still contains systematic SDPA > Sage and 30-step > reduced-step patterns
that the BTRM head can learn from. The smoke test validates the pipeline,
not the quality of the learned reward model.

---

## 8. Implementation Steps

### Phase A: Trajectory loader + offline BTRM test

1. Write `test_e2e_real_trajectories.py` with:
   - Manifest loading
   - Pair construction (scrimble + scrongle)
   - BTRM training loop using `train_btrm_step` endpoint
   - Logging: loss, per_head_accuracy, per-head breakdown
2. Run against live server
3. Verify: both heads active, loss decreasing, no NaN

### Phase B: Policy optimization with server-side gradients

4. Add policy optimization phase to the same script
5. Uses live rollouts (not pre-computed trajectories)
6. Calls training endpoints: `score_btrm`, `accumulate_policy_gradients`,
   `policy_optimizer_step`
7. Run 10 iterations
8. Verify: gradients flow, weights update, no NaN

### Phase C: Round-trip verification

9. Before/after weight push, call `get_lora_state_dict` and verify
   tensors actually changed
10. Compare rollout outputs before and after 10 policy iterations
    (cosine similarity < 1.0 confirms weights affect generation)

---

## 9. Estimated Resource Requirements

| Phase | Server VRAM | Client VRAM | Wall Time (est.) |
|-------|------------|-------------|------------------|
| BTRM 10 steps | ~8GB (FP8 diff + TE for encode + BTRM head) | ~0 (pure scheduling) | ~60s |
| Policy 10 iters (K=2, 10 euler steps) | ~10GB (FP8 + LoRA + grad + optimizers) | ~0 (pure scheduling) | ~5min |

Total: ~6 minutes on a single 4090, sequential phases.

---

## 10. Script Invocation

```bash
# Start server (separate terminal)
.venv/Scripts/python.exe -m futudiffu.server --port 5555 \
    --fp8-diff F:\dox\ai\comfyui\ComfyUI\models\diffusion_models\z_image_fp8_blockwise.safetensors \
    --te F:\dox\ai\comfyui\ComfyUI\models\text_encoders\qwen_3_4b.safetensors \
    --vae F:\dox\ai\comfyui\ComfyUI\models\vae\zimage.safetensors

# Run test
.venv/Scripts/python.exe test_e2e_real_trajectories.py \
    --port 5555 \
    --dataset F:\dox\repos\ai\futudiffu\btrm_dataset \
    --btrm-steps 10 \
    --policy-steps 10 \
    --rollout-steps 10 \
    --group-size 2
```

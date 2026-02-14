"""Preparedness exercise: LoRA save/load/scale roundtrip.

Steps:
  1. Inject rtheta, record random init weights
  2. Perturb weights (simulate 1 training step), confirm diff from init
  3. Run at scale=0 and scale=1 in same batch, confirm different outputs
  4. Save rtheta weights to disk
  5. Kill server (caller does this manually between phases)
  6. Load rtheta onto fresh server, verify same outputs as step 3

Usage:
  # Phase 1: server running, run steps 1-4
  .venv/Scripts/python.exe test_lora_save_load.py --port 5555 --phase 1

  # (kill and restart server)

  # Phase 2: fresh server, run steps 5-6
  .venv/Scripts/python.exe test_lora_save_load.py --port 5555 --phase 2
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from futudiffu.client import InferenceClient

WORK_DIR = Path(__file__).parent / "lora_roundtrip_test"
WEIGHTS_FILE = WORK_DIR / "rtheta_trained.safetensors"
REFERENCE_FILE = WORK_DIR / "reference_outputs.pt"
SEED = 42
N_STEPS = 4
CFG = 4.0
WIDTH = 512
HEIGHT = 512
PROMPT = "a red cube on a white table"


def phase1(client):
    """Inject, train 1 step, test scale 0 vs 1, save."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    # --- Encode prompts FIRST (before inject_lora) ---
    # The server lifecycle frees diffusion when TE loads, and vice versa.
    # Encoding must happen before LoRA injection or the reload wipes LoRAs.
    print("\n[0] Encode prompts (before LoRA injection)")
    pos_cond = client.encode_prompt(PROMPT)
    neg_cond = client.encode_prompt("")
    print(f"    pos_cond: {pos_cond.shape}, neg_cond: {neg_cond.shape}")

    # --- Step 1: Inject rtheta on layers 28-29, record init ---
    print("\n[1] Inject rtheta (layers 28-29, rank=8, alpha=16)")
    n = client.inject_lora("rtheta", rank=8, alpha=16.0, layer_indices=[28, 29])
    print(f"    {n} adapters injected")

    init_sd = client.get_lora_state_dict("rtheta")
    print(f"    {len(init_sd)} init weight tensors retrieved")

    # Sanity: B should be zeros at init
    b_keys = [k for k in init_sd if k.endswith(".lora_B")]
    all_zero = all(init_sd[k].abs().max().item() == 0.0 for k in b_keys)
    print(f"    All lora_B zero at init: {all_zero}")
    assert all_zero, "lora_B should be zero-initialized"

    # --- Step 2: Perturb weights (simulate 1 training step) ---
    print("\n[2] Perturb weights (simulate 1 optimizer step)")
    trained_sd = {}
    for k, v in init_sd.items():
        if k.endswith(".lora_B"):
            trained_sd[k] = v + torch.randn_like(v) * 0.01
        else:
            trained_sd[k] = v.clone()

    client.update_lora_weights(trained_sd)
    readback = client.get_lora_state_dict("rtheta")

    diffs = []
    for k in b_keys:
        d = (readback[k] - init_sd[k]).abs().max().item()
        diffs.append(d)
    print(f"    Max lora_B diff from init: {max(diffs):.6f}")
    assert max(diffs) > 0.001, "Weights should differ after perturbation"
    print("    PASS: weights differ from random init")

    # --- Step 3: Scale 0 vs 1 ---
    print("\n[3] Test scale=0 vs scale=1")

    # Scale = 1.0 (adapter active)
    client.set_adapter_config("rtheta", scale=1.0)
    result_on = client.sample_trajectory(
        pos_cond=pos_cond, neg_cond=neg_cond, seed=SEED,
        n_steps=N_STEPS, cfg=CFG, width=WIDTH, height=HEIGHT,
        attention_backend="sdpa",
    )
    latent_on = result_on["final"]
    print(f"    scale=1 final: mean={latent_on.float().mean():.6f}, "
          f"std={latent_on.float().std():.6f}")

    # Scale = 0.0 (adapter off)
    client.set_adapter_config("rtheta", scale=0.0)
    result_off = client.sample_trajectory(
        pos_cond=pos_cond, neg_cond=neg_cond, seed=SEED,
        n_steps=N_STEPS, cfg=CFG, width=WIDTH, height=HEIGHT,
        attention_backend="sdpa",
    )
    latent_off = result_off["final"]
    print(f"    scale=0 final: mean={latent_off.float().mean():.6f}, "
          f"std={latent_off.float().std():.6f}")

    # Compare
    cos = torch.nn.functional.cosine_similarity(
        latent_on.float().flatten().unsqueeze(0),
        latent_off.float().flatten().unsqueeze(0),
    ).item()
    diff = (latent_on.float() - latent_off.float()).abs().mean().item()
    print(f"    cosine(on, off) = {cos:.6f}")
    print(f"    mean abs diff   = {diff:.6f}")
    assert cos < 0.9999, f"scale=0 and scale=1 should produce different outputs (cos={cos})"
    print("    PASS: scale=0 and scale=1 produce different outputs")

    # --- Step 4: Save weights and reference outputs ---
    print("\n[4] Save weights and reference outputs")
    save_file(trained_sd, str(WEIGHTS_FILE))
    print(f"    Weights saved to {WEIGHTS_FILE} ({WEIGHTS_FILE.stat().st_size} bytes)")

    torch.save({
        "latent_on": latent_on.cpu(),
        "latent_off": latent_off.cpu(),
        "cos": cos,
    }, str(REFERENCE_FILE))
    print(f"    Reference outputs saved to {REFERENCE_FILE}")

    print("\n" + "=" * 60)
    print("  Phase 1 complete. Now:")
    print("  1. Kill the server")
    print("  2. Restart a fresh server")
    print("  3. Run: test_lora_save_load.py --phase 2")
    print("=" * 60)


def phase2(client):
    """Inject fresh rtheta, load saved weights, verify outputs match."""
    assert WEIGHTS_FILE.exists(), f"No weights at {WEIGHTS_FILE} — run phase 1 first"
    assert REFERENCE_FILE.exists(), f"No reference at {REFERENCE_FILE} — run phase 1 first"

    ref = torch.load(str(REFERENCE_FILE), map_location="cpu", weights_only=True)
    ref_on = ref["latent_on"]
    ref_off = ref["latent_off"]
    ref_cos = ref["cos"]

    # --- Encode prompts FIRST (before inject_lora) ---
    print("\n[5a] Encode prompts (before LoRA injection)")
    pos_cond = client.encode_prompt(PROMPT)
    neg_cond = client.encode_prompt("")

    # --- Step 5: Inject fresh rtheta (zeros), load saved weights ---
    print("\n[5b] Inject fresh rtheta + load saved weights")
    n = client.inject_lora("rtheta", rank=8, alpha=16.0, layer_indices=[28, 29])
    print(f"    {n} adapters injected (fresh zeros)")

    sd = load_file(str(WEIGHTS_FILE))
    print(f"    Loading {len(sd)} weight tensors from {WEIGHTS_FILE}")
    client.update_lora_weights(sd)

    # Readback and verify against saved
    readback = client.get_lora_state_dict("rtheta")
    max_load_err = 0.0
    for k in sd:
        err = (readback[k] - sd[k]).abs().max().item()
        max_load_err = max(max_load_err, err)
    print(f"    Max load error (readback vs file): {max_load_err:.8f}")
    assert max_load_err < 1e-4, f"Load error too large: {max_load_err}"

    # --- Step 6: Verify same outputs ---
    print("\n[6] Verify outputs match phase 1")

    # Scale = 1.0
    client.set_adapter_config("rtheta", scale=1.0)
    result_on = client.sample_trajectory(
        pos_cond=pos_cond, neg_cond=neg_cond, seed=SEED,
        n_steps=N_STEPS, cfg=CFG, width=WIDTH, height=HEIGHT,
        attention_backend="sdpa",
    )
    latent_on = result_on["final"]

    cos_on = torch.nn.functional.cosine_similarity(
        latent_on.float().flatten().unsqueeze(0),
        ref_on.float().flatten().unsqueeze(0),
    ).item()
    diff_on = (latent_on.float() - ref_on.float()).abs().mean().item()
    print(f"    scale=1: cosine(new, ref) = {cos_on:.6f}, mean_diff = {diff_on:.8f}")

    # Scale = 0.0
    client.set_adapter_config("rtheta", scale=0.0)
    result_off = client.sample_trajectory(
        pos_cond=pos_cond, neg_cond=neg_cond, seed=SEED,
        n_steps=N_STEPS, cfg=CFG, width=WIDTH, height=HEIGHT,
        attention_backend="sdpa",
    )
    latent_off = result_off["final"]

    cos_off = torch.nn.functional.cosine_similarity(
        latent_off.float().flatten().unsqueeze(0),
        ref_off.float().flatten().unsqueeze(0),
    ).item()
    diff_off = (latent_off.float() - ref_off.float()).abs().mean().item()
    print(f"    scale=0: cosine(new, ref) = {cos_off:.6f}, mean_diff = {diff_off:.8f}")

    # Also re-verify on vs off still differ
    cos_on_off = torch.nn.functional.cosine_similarity(
        latent_on.float().flatten().unsqueeze(0),
        latent_off.float().flatten().unsqueeze(0),
    ).item()
    print(f"    cosine(on, off) this run = {cos_on_off:.6f} (phase 1 was {ref_cos:.6f})")

    print("\n" + "=" * 60)
    all_pass = True
    if cos_on > 0.9999:
        print("  PASS: scale=1 output matches across save/load")
    else:
        print(f"  FAIL: scale=1 output diverged (cos={cos_on:.6f})")
        all_pass = False

    if cos_off > 0.9999:
        print("  PASS: scale=0 output matches across save/load")
    else:
        print(f"  FAIL: scale=0 output diverged (cos={cos_off:.6f})")
        all_pass = False

    if cos_on_off < 0.9999:
        print("  PASS: scale=0 vs scale=1 still differ after reload")
    else:
        print(f"  FAIL: scale=0 vs scale=1 identical (cos={cos_on_off:.6f})")
        all_pass = False

    if all_pass:
        print("\n  ALL TESTS PASSED")
    else:
        print("\n  SOME TESTS FAILED")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2])
    args = parser.parse_args()

    client = InferenceClient(f"tcp://localhost:{args.port}")
    status = client.status()
    print(f"Server status: {status}")

    if args.phase == 1:
        phase1(client)
    else:
        phase2(client)


if __name__ == "__main__":
    main()

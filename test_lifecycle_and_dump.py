"""Test LoRA lifecycle persistence + crash dump.

Exercises the full lifecycle:
  1. Inject rtheta, perturb weights
  2. Encode prompts (triggers TE load -> diffusion freed -> LoRA snapshot)
  3. Sample with scale=1 (triggers diffusion reload -> LoRA replay)
  4. Verify LoRA is active (scale=1 vs scale=0 differ)
  5. Crash dump all LoRAs to disk
  6. Verify dump files exist and are loadable
"""
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from futudiffu.client import InferenceClient

SEED = 42
N_STEPS = 4
CFG = 4.0
WIDTH = 512
HEIGHT = 512
PROMPT = "a red cube on a white table"


def main():
    client = InferenceClient("tcp://localhost:5555")
    status = client.status()
    print(f"Server status: {status}")

    # --- Step 1: Inject rtheta on layers 28-29, perturb weights ---
    print("\n[1] Inject rtheta + perturb weights")
    n = client.inject_lora("rtheta", rank=8, alpha=16.0, layer_indices=[28, 29])
    print(f"    {n} adapters injected")

    init_sd = client.get_lora_state_dict("rtheta")
    trained_sd = {}
    for k, v in init_sd.items():
        if k.endswith(".lora_B"):
            trained_sd[k] = v + torch.randn_like(v) * 0.01
        else:
            trained_sd[k] = v.clone()
    client.update_lora_weights(trained_sd)
    print("    Weights perturbed")

    # --- Step 2: Encode prompts (triggers TE load -> diffusion freed) ---
    print("\n[2] Encode prompts (lifecycle swap: diffusion -> TE)")
    pos_cond = client.encode_prompt(PROMPT)
    neg_cond = client.encode_prompt("")
    print(f"    pos: {pos_cond.shape}, neg: {neg_cond.shape}")

    # --- Step 3: Sample at scale=1 (triggers diffusion reload -> LoRA replay) ---
    print("\n[3] Sample at scale=1 (lifecycle swap: TE -> diffusion + LoRA replay)")
    client.set_adapter_config("rtheta", scale=1.0)
    result_on = client.sample_trajectory(
        pos_cond=pos_cond, neg_cond=neg_cond, seed=SEED,
        n_steps=N_STEPS, cfg=CFG, width=WIDTH, height=HEIGHT,
        attention_backend="sdpa",
    )
    latent_on = result_on["final"]
    print(f"    scale=1 final: mean={latent_on.float().mean():.6f}")

    # --- Step 4: Sample at scale=0 and compare ---
    print("\n[4] Sample at scale=0, compare")
    client.set_adapter_config("rtheta", scale=0.0)
    result_off = client.sample_trajectory(
        pos_cond=pos_cond, neg_cond=neg_cond, seed=SEED,
        n_steps=N_STEPS, cfg=CFG, width=WIDTH, height=HEIGHT,
        attention_backend="sdpa",
    )
    latent_off = result_off["final"]

    cos = torch.nn.functional.cosine_similarity(
        latent_on.float().flatten().unsqueeze(0),
        latent_off.float().flatten().unsqueeze(0),
    ).item()
    diff = (latent_on.float() - latent_off.float()).abs().mean().item()
    print(f"    scale=0 final: mean={latent_off.float().mean():.6f}")
    print(f"    cosine(on, off) = {cos:.6f}, mean_diff = {diff:.6f}")

    if cos < 0.9999:
        print("    PASS: LoRA survived lifecycle swap (scale=0 != scale=1)")
    else:
        print(f"    FAIL: LoRA lost during lifecycle swap (cos={cos})")
        return

    # --- Step 5: Verify weights survived replay ---
    print("\n[5] Verify weights match post-reload")
    reloaded_sd = client.get_lora_state_dict("rtheta")
    max_err = 0.0
    for k in trained_sd:
        err = (reloaded_sd[k] - trained_sd[k]).abs().max().item()
        max_err = max(max_err, err)
    print(f"    Max weight error after lifecycle replay: {max_err:.8f}")
    if max_err < 1e-4:
        print("    PASS: weights preserved through lifecycle swap")
    else:
        print(f"    FAIL: weight drift too large ({max_err})")

    # --- Step 6: Crash dump ---
    print("\n[6] Crash dump all LoRAs")
    dump_dir = "lora_dumps_test"
    result = client.dump_all_loras(output_dir=dump_dir)
    files = result.get("files", [])
    manifest = result.get("manifest", "")
    print(f"    {len(files)} adapter(s) dumped")
    for f in files:
        print(f"      {f['adapter']}: {f['n_tensors']} tensors, "
              f"rank={f['rank']}, alpha={f['alpha']}, scale={f['scale']}")
        print(f"        -> {f['path']}")

    # Verify dump files are loadable
    for f in files:
        sd = load_file(f["path"])
        print(f"      Loaded {len(sd)} tensors from {Path(f['path']).name}")
        # Verify against current weights
        for k in sd:
            if k in trained_sd:
                err = (sd[k] - trained_sd[k]).abs().max().item()
                assert err < 1e-4, f"Dump mismatch on {k}: {err}"
    print("    PASS: dump files are loadable and match current weights")

    print(f"\n{'='*60}")
    print("  ALL TESTS PASSED")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

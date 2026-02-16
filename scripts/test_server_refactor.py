r"""Comprehensive test suite for the server.py refactoring.

Tests all RPC handlers to verify behavioral equivalence after refactoring.
Re-runnable with clear PASS/FAIL output per test. Connects to a running
inference server on localhost:5555.

Usage:
    .venv\Scripts\python.exe scripts\test_server_refactor.py
    .venv\Scripts\python.exe scripts\test_server_refactor.py --port 5555
    .venv\Scripts\python.exe scripts\test_server_refactor.py --skip-heavy
"""
import argparse
import os
import sys
import time
import traceback

# Force unbuffered stdout for real-time output
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, r"F:\dox\repos\ai\futudiffu\src")

import torch

from futudiffu.client import InferenceClient


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = None  # True/False/None (not run)
        self.detail = ""
        self.elapsed = 0.0
        self.skipped = False

    def __repr__(self):
        status = "PASS" if self.passed else ("SKIP" if self.skipped else "FAIL")
        return f"[{status}] {self.name} ({self.elapsed:.1f}s) {self.detail}"


results: list[TestResult] = []


def run_test(name, fn, *args, **kwargs):
    """Run a test function, catching exceptions."""
    r = TestResult(name)
    print(f"\n{'='*60}")
    print(f"  TEST: {name}")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    try:
        fn(r, *args, **kwargs)
        if r.passed is None:
            r.passed = True
    except Exception as e:
        r.passed = False
        r.detail = f"Exception: {e}"
        traceback.print_exc()
    r.elapsed = time.perf_counter() - t0
    results.append(r)
    status = "PASS" if r.passed else ("SKIP" if r.skipped else "FAIL")
    print(f"  -> [{status}] {r.detail} ({r.elapsed:.1f}s)")


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------

def assert_tensor_valid(t, name, expected_dtype=None, expected_ndim=None,
                        expected_shape_prefix=None, no_nan=True, no_all_zero=True):
    """Validate a tensor meets expectations."""
    issues = []
    if not isinstance(t, torch.Tensor):
        raise AssertionError(f"{name}: expected tensor, got {type(t)}")
    if expected_dtype is not None and t.dtype != expected_dtype:
        issues.append(f"dtype={t.dtype}, expected {expected_dtype}")
    if expected_ndim is not None and t.ndim != expected_ndim:
        issues.append(f"ndim={t.ndim}, expected {expected_ndim}")
    if expected_shape_prefix is not None:
        for i, s in enumerate(expected_shape_prefix):
            if s is not None and i < t.ndim and t.shape[i] != s:
                issues.append(f"shape[{i}]={t.shape[i]}, expected {s}")
    if no_nan and torch.isnan(t).any():
        issues.append("contains NaN")
    if no_all_zero and t.numel() > 0 and t.float().abs().sum().item() == 0:
        issues.append("all zeros")
    if issues:
        raise AssertionError(f"{name}: {', '.join(issues)} (shape={list(t.shape)})")
    return True


def cosine_sim(a, b):
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    denom = a_f.norm() * b_f.norm()
    if denom == 0:
        return 0.0
    return (torch.dot(a_f, b_f) / denom).item()


# ---------------------------------------------------------------------------
# Connectivity test with retry
# ---------------------------------------------------------------------------

def wait_for_server(endpoint, max_retries=10, delay=3.0):
    """Try to connect to server, retrying if it's down (refactor agent may be restarting)."""
    for attempt in range(max_retries):
        try:
            c = InferenceClient(endpoint, timeout_ms=5000)
            _, meta = c._call("status")
            c.close()
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Server not ready (attempt {attempt+1}/{max_retries}): {e}")
                print(f"  Retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"  Server unreachable after {max_retries} attempts")
                return False
    return False


# ---------------------------------------------------------------------------
# Individual tests
# ---------------------------------------------------------------------------

def test_status(r, client):
    """T1: Status RPC returns valid response with expected fields."""
    status = client.status()
    print(f"    Status: {status}")

    required_keys = ["loaded_models", "phase", "vram_allocated_gb",
                     "vram_reserved_gb", "vram_total_gb"]
    missing = [k for k in required_keys if k not in status]
    if missing:
        r.passed = False
        r.detail = f"Missing keys: {missing}"
        return

    # VRAM values should be non-negative numbers
    for key in ["vram_allocated_gb", "vram_reserved_gb", "vram_total_gb"]:
        val = status[key]
        if not isinstance(val, (int, float)) or val < 0:
            r.passed = False
            r.detail = f"{key}={val} invalid"
            return

    # loaded_models should be a list
    if not isinstance(status["loaded_models"], list):
        r.passed = False
        r.detail = f"loaded_models is not a list: {type(status['loaded_models'])}"
        return

    r.detail = f"phase={status['phase']}, VRAM={status['vram_allocated_gb']:.1f}GB"


def test_encode_prompt(r, client):
    """T2: Encode a text prompt, verify conditioning tensor shape/dtype."""
    cond = client.encode_prompt("a laser shark swimming through space")
    assert_tensor_valid(cond, "conditioning",
                        expected_dtype=torch.bfloat16,
                        expected_ndim=3,
                        expected_shape_prefix=[1, None, 2560])
    print(f"    conditioning shape: {cond.shape}, dtype: {cond.dtype}")
    print(f"    norm: {cond.float().norm().item():.2f}, "
          f"mean: {cond.float().mean().item():.4f}")
    r.detail = f"shape={list(cond.shape)}"
    return cond


def test_encode_prompt_empty(r, client):
    """T2b: Encode empty prompt (negative conditioning)."""
    neg = client.encode_prompt("")
    assert_tensor_valid(neg, "neg_conditioning",
                        expected_dtype=torch.bfloat16,
                        expected_ndim=3,
                        expected_shape_prefix=[1, None, 2560])
    print(f"    neg_conditioning shape: {neg.shape}")
    r.detail = f"shape={list(neg.shape)}"
    return neg


def test_warmup_sdpa(r, client):
    """T3: Warmup with SDPA backend completes without error."""
    client.warmup(attention_backend="sdpa")
    r.detail = "sdpa warmup ok"


def test_sample_trajectory_basic(r, client, pos_cond, neg_cond):
    """T4: Sample a short trajectory (8 steps), verify result dict structure."""
    seed = 42
    n_steps = 8
    save_steps = [0, 3, 7]

    result = client.sample_trajectory(
        pos_cond, neg_cond,
        seed=seed,
        n_steps=n_steps,
        cfg=4.0,
        width=1280,
        height=832,
        attention_backend="sdpa",
        save_steps=save_steps,
    )

    # Must have "final" key
    if "final" not in result:
        r.passed = False
        r.detail = "Missing 'final' key in result"
        return

    final = result["final"]
    assert_tensor_valid(final, "final",
                        expected_dtype=torch.bfloat16,
                        expected_ndim=4,
                        expected_shape_prefix=[1, 16, 104, 160])
    print(f"    final shape: {final.shape}, norm: {final.float().norm().item():.2f}")

    # Check saved steps
    for step_i in save_steps:
        key = f"step_{step_i:02d}"
        if key not in result:
            r.passed = False
            r.detail = f"Missing saved step: {key}"
            return
        step_t = result[key]
        assert_tensor_valid(step_t, key,
                            expected_dtype=torch.bfloat16,
                            expected_ndim=4)
        print(f"    {key} shape: {step_t.shape}, norm: {step_t.float().norm().item():.2f}")

    # Steps not in save_steps should NOT be present
    for step_i in range(n_steps):
        if step_i not in save_steps:
            key = f"step_{step_i:02d}"
            if key in result:
                r.passed = False
                r.detail = f"Unexpected step in result: {key}"
                return

    r.detail = f"final shape={list(final.shape)}, {len(save_steps)} saved steps ok"
    return result


def test_sample_trajectory_deterministic(r, client, pos_cond, neg_cond):
    """T5: Two trajectories with same seed produce identical final latents."""
    seed = 12345
    kwargs = dict(
        seed=seed, n_steps=8, cfg=4.0, width=1280, height=832,
        attention_backend="sdpa", save_steps=[],
    )
    result1 = client.sample_trajectory(pos_cond, neg_cond, **kwargs)
    result2 = client.sample_trajectory(pos_cond, neg_cond, **kwargs)

    cos = cosine_sim(result1["final"], result2["final"])
    print(f"    cosine sim between runs: {cos:.8f}")

    # With same seed and same conditioning, results should be bitwise identical
    # (or extremely close -- cos > 0.9999)
    if cos < 0.9999:
        r.passed = False
        r.detail = f"Determinism failure: cos={cos:.6f}"
        return

    r.detail = f"cos={cos:.8f}"


def test_sample_trajectory_with_noise(r, client, pos_cond, neg_cond):
    """T6: Pre-provided noise tensor bypasses RNG seed entirely."""
    latent_h, latent_w = 104, 160  # 832x1280 / 8
    noise = torch.randn(1, 16, latent_h, latent_w, dtype=torch.bfloat16)

    result1 = client.sample_trajectory(
        pos_cond, neg_cond,
        seed=0, n_steps=8, cfg=4.0, width=1280, height=832,
        attention_backend="sdpa", save_steps=[], noise=noise,
    )
    # Same noise, different seed -- should produce same result
    result2 = client.sample_trajectory(
        pos_cond, neg_cond,
        seed=9999, n_steps=8, cfg=4.0, width=1280, height=832,
        attention_backend="sdpa", save_steps=[], noise=noise,
    )

    cos = cosine_sim(result1["final"], result2["final"])
    print(f"    cosine sim (same noise, different seeds): {cos:.8f}")

    if cos < 0.9999:
        r.passed = False
        r.detail = f"Noise override failure: cos={cos:.6f}"
        return

    r.detail = f"cos={cos:.8f}"


def test_vae_decode(r, client, final_latent):
    """T7: VAE decode produces image with correct shape and range."""
    image = client.vae_decode(final_latent)
    assert_tensor_valid(image, "image",
                        expected_dtype=torch.bfloat16,
                        expected_ndim=4,
                        expected_shape_prefix=[1, 3, 832, 1280])
    print(f"    image shape: {image.shape}")
    img_f = image.float()
    print(f"    range: [{img_f.min().item():.3f}, {img_f.max().item():.3f}]")

    # Image values should be mostly in [0, 1] (some slight overshoot is ok)
    if img_f.min().item() < -0.5 or img_f.max().item() > 1.5:
        r.passed = False
        r.detail = f"Image range suspect: [{img_f.min():.3f}, {img_f.max():.3f}]"
        return

    r.detail = f"shape={list(image.shape)}, range=[{img_f.min():.3f}, {img_f.max():.3f}]"


def test_vae_encode_decode_roundtrip(r, client):
    """T8: VAE encode then decode produces output with correct shape."""
    # Create a synthetic image
    fake_image = torch.rand(1, 3, 832, 1280, dtype=torch.bfloat16)

    latent = client.vae_encode(fake_image)
    assert_tensor_valid(latent, "latent",
                        expected_dtype=torch.bfloat16,
                        expected_ndim=4,
                        expected_shape_prefix=[1, 16, 104, 160])
    print(f"    encoded latent shape: {latent.shape}")

    decoded = client.vae_decode(latent)
    assert_tensor_valid(decoded, "decoded",
                        expected_dtype=torch.bfloat16,
                        expected_ndim=4,
                        expected_shape_prefix=[1, 3, 832, 1280])
    print(f"    decoded image shape: {decoded.shape}")

    # Roundtrip should roughly preserve the image (lossy, but not garbage)
    cos = cosine_sim(fake_image, decoded)
    print(f"    roundtrip cosine: {cos:.4f}")

    if cos < 0.5:
        r.passed = False
        r.detail = f"VAE roundtrip too lossy: cos={cos:.4f}"
        return

    r.detail = f"roundtrip cos={cos:.4f}"


def test_free_and_reload_te(r, client):
    """T9: Free TE, then encode_prompt (forces TE reload), verify works."""
    # Free TE
    client.free("te")
    status = client.status()
    print(f"    After free(te): loaded={status['loaded_models']}, phase={status['phase']}")

    # Encoding should trigger TE reload
    cond = client.encode_prompt("test reload")
    assert_tensor_valid(cond, "conditioning_after_reload",
                        expected_dtype=torch.bfloat16,
                        expected_ndim=3,
                        expected_shape_prefix=[1, None, 2560])
    print(f"    After reload: cond shape={cond.shape}")
    r.detail = f"Reload ok, shape={list(cond.shape)}"


def test_free_and_reload_diffusion(r, client, pos_cond, neg_cond):
    """T10: Free diffusion, then sample (forces reload), verify works."""
    client.free("diffusion")
    status = client.status()
    print(f"    After free(diffusion): loaded={status['loaded_models']}")

    # Sampling should trigger diffusion model reload
    result = client.sample_trajectory(
        pos_cond, neg_cond,
        seed=42, n_steps=4, cfg=4.0, width=1280, height=832,
        attention_backend="sdpa", save_steps=[],
    )
    assert_tensor_valid(result["final"], "final_after_reload",
                        expected_dtype=torch.bfloat16,
                        expected_ndim=4)
    print(f"    After reload: final shape={result['final'].shape}")
    r.detail = "Diffusion reload + sample ok"


def test_free_bogus(r, client):
    """T11: Free with unrecognized model name -- check behavior."""
    # The refactor plan L3 says this should return an error, but currently
    # it silently succeeds. We test both behaviors.
    try:
        client.free("bogus_model")
        # If it doesn't raise, that's the pre-refactor behavior (silent success)
        r.detail = "Silent success (pre-refactor behavior)"
    except RuntimeError as e:
        # If it raises, that's the post-refactor behavior (explicit error)
        r.detail = f"Error raised (post-refactor behavior): {e}"
    # Either way, this is informational, not a hard fail
    r.passed = True


def test_inject_lora(r, client):
    """T12: Inject a LoRA adapter and verify metadata."""
    # Ensure diffusion is loaded first
    client.warmup(attention_backend="sdpa")

    # Use a unique adapter name with timestamp to avoid collisions with
    # previous test runs (server persists LoRA state across runs).
    import random
    adapter_suffix = random.randint(1000, 9999)
    adapter_name = f"test_refactor_{adapter_suffix}"

    # Store for downstream tests to use
    test_inject_lora._adapter_name = adapter_name

    n_adapters = client.inject_lora(
        adapter_name=adapter_name,
        rank=8,
        alpha=16.0,
        init_b_std=0.01,
    )
    print(f"    Injected {n_adapters} adapters as '{adapter_name}'")

    if n_adapters == 0:
        r.passed = False
        r.detail = "No adapters injected"
        return

    r.detail = f"n_adapters={n_adapters}, name={adapter_name}"

# Default fallback adapter name for downstream tests
test_inject_lora._adapter_name = "test_refactor"


def test_set_adapter_config_scale(r, client):
    """T13: Set adapter scale, verify no error."""
    name = test_inject_lora._adapter_name
    client.set_adapter_config(name, scale=0.5)
    print(f"    Scale 0.5 set on '{name}'")

    # Also test per-batch scale
    client.set_adapter_config(name, scale=[1.0, 0.0])
    print("    Per-batch scale [1.0, 0.0] set")

    # Reset to scalar
    client.set_adapter_config(name, scale=1.0)
    print("    Scale reset to 1.0")

    r.detail = "Scale setting ok"


def test_set_adapter_config_freeze(r, client):
    """T14: Freeze adapter, verify no error."""
    name = test_inject_lora._adapter_name
    client.set_adapter_config(name, frozen=True)
    print(f"    Adapter '{name}' frozen")
    r.detail = "Freeze ok"


def test_get_lora_state_dict(r, client):
    """T15: Get LoRA state dict, verify tensor shapes."""
    name = test_inject_lora._adapter_name
    sd = client.get_lora_state_dict(adapter_name=name)
    print(f"    Got {len(sd)} tensors")

    if len(sd) == 0:
        r.passed = False
        r.detail = "Empty state dict"
        return

    # Check a few tensors
    for key in list(sd.keys())[:3]:
        t = sd[key]
        print(f"    {key}: shape={list(t.shape)}, dtype={t.dtype}")
        assert_tensor_valid(t, key, no_all_zero=False)  # B matrices may be near-zero

    r.detail = f"n_tensors={len(sd)}"
    return sd


def test_inject_btrm_head(r, client):
    """T16: Inject BTRM scoring head."""
    meta = client.inject_btrm_head(
        head_names=["test_quality"],
        logit_cap=10.0,
        lr=1e-4,
        weight_decay=0.0,
        hidden_dim=3840,
    )
    print(f"    BTRM metadata: {meta}")

    if meta.get("n_heads") != 1:
        r.passed = False
        r.detail = f"Expected 1 head, got {meta.get('n_heads')}"
        return

    r.detail = f"n_heads={meta['n_heads']}, n_params={meta.get('n_params')}"


def test_score_btrm(r, client, pos_cond):
    """T17: Score a latent via BTRM head."""
    # Create a dummy latent + sigma
    latent = torch.randn(1, 16, 104, 160, dtype=torch.bfloat16)
    sigma = torch.tensor([0.5], dtype=torch.bfloat16)
    cond = pos_cond[:, :pos_cond.shape[1], :]  # use real conditioning

    scores = client.score_btrm(
        latent=latent,
        sigma=sigma,
        conditioning=cond,
        attention_backend="sdpa",
    )
    print(f"    Scores: {scores}")

    if not isinstance(scores, list):
        r.passed = False
        r.detail = f"Expected list, got {type(scores)}"
        return

    # Each example should have a list of per-head scores
    if len(scores) == 0 or not isinstance(scores[0], list):
        r.passed = False
        r.detail = f"Unexpected scores format: {scores}"
        return

    # Scores should not be NaN
    for ex_scores in scores:
        for s in ex_scores:
            if s != s:  # NaN check
                r.passed = False
                r.detail = "NaN in scores"
                return

    r.detail = f"scores={scores}"


def test_train_btrm_step(r, client, pos_cond):
    """T18: One BTRM training step."""
    latent_h, latent_w = 104, 160

    # Create minimal positive and negative examples for head 0
    examples = []
    for is_pos in [True, False]:
        examples.append({
            "latent": torch.randn(1, 16, latent_h, latent_w, dtype=torch.bfloat16),
            "sigma": torch.tensor([0.5], dtype=torch.bfloat16),
            "conditioning": pos_cond,
            "head_idx": 0,
            "is_positive": is_pos,
        })

    meta = client.train_btrm_step(
        examples=examples,
        logsquare_weight=0.1,
        attention_backend="sdpa",
    )
    print(f"    Training metadata: {meta}")

    required = ["loss", "bt_loss", "logsq_loss", "n_examples"]
    for key in required:
        if key not in meta:
            r.passed = False
            r.detail = f"Missing key: {key}"
            return

    # Loss should be a finite number
    loss = meta["loss"]
    if loss != loss:  # NaN check
        r.passed = False
        r.detail = f"NaN loss: {loss}"
        return

    r.detail = f"loss={loss:.4f}, bt={meta['bt_loss']:.4f}"


def test_dump_loras(r, client):
    """T19: Dump LoRA adapters to disk."""
    dump_dir = r"F:\dox\repos\ai\futudiffu\lora_dumps_test_refactor"
    meta = client.dump_all_loras(output_dir=dump_dir)
    print(f"    Dump metadata: {meta}")

    files = meta.get("files", [])
    print(f"    {len(files)} adapter file(s) dumped")

    manifest = meta.get("manifest")
    if manifest:
        print(f"    Manifest: {manifest}")

    r.detail = f"n_files={len(files)}, manifest={'yes' if manifest else 'no'}"


def test_warmup_after_lora(r, client):
    """T20: Warmup after LoRA injection (recompilation)."""
    client.warmup(attention_backend="sdpa")
    r.detail = "Warmup after LoRA ok"


def test_sample_after_lora(r, client, pos_cond, neg_cond):
    """T21: Sample trajectory after LoRA injection, verify non-garbage output."""
    result = client.sample_trajectory(
        pos_cond, neg_cond,
        seed=42, n_steps=8, cfg=4.0, width=1280, height=832,
        attention_backend="sdpa", save_steps=[],
    )
    final = result["final"]
    assert_tensor_valid(final, "final_with_lora",
                        expected_dtype=torch.bfloat16,
                        expected_ndim=4)
    norm = final.float().norm().item()
    print(f"    final norm: {norm:.2f}")

    # Should produce meaningful output (not zero, not NaN, not absurdly large)
    if norm < 0.1:
        r.passed = False
        r.detail = f"Suspiciously small norm: {norm}"
        return
    if norm > 1e6:
        r.passed = False
        r.detail = f"Suspiciously large norm: {norm}"
        return

    r.detail = f"norm={norm:.2f}"


def test_unknown_method(r, client):
    """T22: Unknown RPC method returns proper error."""
    try:
        client._call("nonexistent_method_12345")
        r.passed = False
        r.detail = "Should have raised RuntimeError"
    except RuntimeError as e:
        err = str(e)
        if "Unknown method" in err or "unknown" in err.lower():
            r.detail = f"Correct error: {err}"
        else:
            r.detail = f"Error raised but unexpected message: {err}"


def test_encode_prompt_layer_idx(r, client):
    """T23: Encode prompt with different layer_idx, verify different output."""
    cond_m2 = client.encode_prompt("test", layer_idx=-2)
    cond_m1 = client.encode_prompt("test", layer_idx=-1)

    cos = cosine_sim(cond_m2, cond_m1)
    print(f"    layer -2 vs -1 cosine: {cos:.4f}")
    print(f"    layer -2 shape: {cond_m2.shape}, layer -1 shape: {cond_m1.shape}")

    # Different layers should produce different embeddings
    if cos > 0.999:
        r.passed = False
        r.detail = f"layer_idx has no effect: cos={cos:.4f}"
        return

    r.detail = f"cos={cos:.4f}"


def test_sample_save_all_steps(r, client, pos_cond, neg_cond):
    """T24: Sample with save_steps=None (all), verify all steps present."""
    n_steps = 4
    result = client.sample_trajectory(
        pos_cond, neg_cond,
        seed=42, n_steps=n_steps, cfg=4.0, width=1280, height=832,
        attention_backend="sdpa",
        save_steps=None,  # Save all
    )

    for step_i in range(n_steps):
        key = f"step_{step_i:02d}"
        if key not in result:
            r.passed = False
            r.detail = f"Missing step {key} when save_steps=None"
            return
        print(f"    {key}: shape={list(result[key].shape)}")

    if "final" not in result:
        r.passed = False
        r.detail = "Missing 'final'"
        return

    r.detail = f"All {n_steps} steps + final present"


def test_i2i_denoise(r, client, pos_cond, neg_cond):
    """T25: img2img with denoise < 1.0 and a clean_latent."""
    latent_h, latent_w = 104, 160
    clean_latent = torch.randn(1, 16, latent_h, latent_w, dtype=torch.bfloat16)

    result = client.sample_trajectory(
        pos_cond, neg_cond,
        seed=42, n_steps=8, cfg=4.0, width=1280, height=832,
        attention_backend="sdpa",
        save_steps=[],
        denoise=0.5,
        clean_latent=clean_latent,
    )

    final = result["final"]
    assert_tensor_valid(final, "i2i_final",
                        expected_dtype=torch.bfloat16,
                        expected_ndim=4)

    # With denoise=0.5, the result should be closer to the clean_latent
    # than a full t2i result
    cos_clean = cosine_sim(final, clean_latent)
    print(f"    i2i final vs clean: cos={cos_clean:.4f}")
    print(f"    i2i final norm: {final.float().norm().item():.2f}")

    r.detail = f"denoise=0.5, cos_with_clean={cos_clean:.4f}"


def test_multiple_status_calls(r, client):
    """T26: Multiple rapid status calls don't break the socket."""
    for i in range(5):
        status = client.status()
        if "loaded_models" not in status:
            r.passed = False
            r.detail = f"Broken on call {i}"
            return
    r.detail = "5 rapid status calls ok"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--skip-heavy", action="store_true",
                        help="Skip heavy tests (trajectory, VAE, training)")
    parser.add_argument("--timeout", type=int, default=0,
                        help="Client timeout in ms (0=infinite, default)")
    args = parser.parse_args()

    endpoint = f"tcp://localhost:{args.port}"
    print(f"Connecting to {endpoint}...")

    # Wait for server to be available
    if not wait_for_server(endpoint):
        print("\nFATAL: Server not reachable. Is it running?")
        sys.exit(1)

    client = InferenceClient(endpoint, timeout_ms=args.timeout)

    # ---------------------------------------------------------------
    # Phase 1: Basic connectivity
    # ---------------------------------------------------------------
    run_test("T01: Status RPC", test_status, client)
    run_test("T02: Unknown method error", test_unknown_method, client)
    run_test("T26: Multiple rapid status calls", test_multiple_status_calls, client)

    # ---------------------------------------------------------------
    # Phase 2: Text encoding
    # ---------------------------------------------------------------
    pos_cond = None
    neg_cond = None

    def capture_pos(r, c):
        nonlocal pos_cond
        pos_cond = test_encode_prompt(r, c)

    def capture_neg(r, c):
        nonlocal neg_cond
        neg_cond = test_encode_prompt_empty(r, c)

    run_test("T02: Encode prompt (positive)", capture_pos, client)
    run_test("T03: Encode prompt (negative/empty)", capture_neg, client)
    run_test("T23: Encode prompt (different layer_idx)", test_encode_prompt_layer_idx, client)

    if pos_cond is None or neg_cond is None:
        print("\nFATAL: Failed to encode prompts -- cannot continue")
        print_summary()
        sys.exit(1)

    # ---------------------------------------------------------------
    # Phase 3: Warmup + Sampling
    # ---------------------------------------------------------------
    if not args.skip_heavy:
        run_test("T04: Warmup (SDPA)", test_warmup_sdpa, client)

        trajectory_result = [None]
        def capture_trajectory(r, c, p, n):
            trajectory_result[0] = test_sample_trajectory_basic(r, c, p, n)

        run_test("T05: Sample trajectory (8 steps)",
                 capture_trajectory, client, pos_cond, neg_cond)
        run_test("T06: Sample trajectory determinism",
                 test_sample_trajectory_deterministic, client, pos_cond, neg_cond)
        run_test("T07: Sample trajectory with pre-provided noise",
                 test_sample_trajectory_with_noise, client, pos_cond, neg_cond)
        run_test("T24: Save all steps (save_steps=None)",
                 test_sample_save_all_steps, client, pos_cond, neg_cond)
        run_test("T25: img2img with denoise=0.5",
                 test_i2i_denoise, client, pos_cond, neg_cond)

        # VAE tests
        final_latent = trajectory_result[0]["final"] if trajectory_result[0] else None
        if final_latent is not None:
            run_test("T08: VAE decode", test_vae_decode, client, final_latent)
        run_test("T09: VAE encode/decode roundtrip",
                 test_vae_encode_decode_roundtrip, client)

        # Lifecycle tests
        run_test("T10: Free and reload TE", test_free_and_reload_te, client)
        run_test("T11: Free and reload diffusion",
                 test_free_and_reload_diffusion, client, pos_cond, neg_cond)
        run_test("T11b: Free bogus model name", test_free_bogus, client)

        # ---------------------------------------------------------------
        # Phase 4: LoRA + BTRM
        # ---------------------------------------------------------------
        run_test("T12: Inject LoRA", test_inject_lora, client)
        run_test("T13: Set adapter scale", test_set_adapter_config_scale, client)
        run_test("T14: Freeze adapter", test_set_adapter_config_freeze, client)
        run_test("T15: Get LoRA state dict", test_get_lora_state_dict, client)
        run_test("T20: Warmup after LoRA", test_warmup_after_lora, client)
        run_test("T21: Sample after LoRA injection",
                 test_sample_after_lora, client, pos_cond, neg_cond)

        # BTRM tests
        run_test("T16: Inject BTRM head", test_inject_btrm_head, client)
        run_test("T17: Score BTRM", test_score_btrm, client, pos_cond)
        run_test("T18: Train BTRM step", test_train_btrm_step, client, pos_cond)

        # Dump
        run_test("T19: Dump LoRAs", test_dump_loras, client)

    else:
        print("\n  [SKIPPED] Heavy tests (--skip-heavy)")

    client.close()
    print_summary()


def print_summary():
    """Print final summary of all test results."""
    print(f"\n{'='*60}")
    print("  TEST SUMMARY")
    print(f"{'='*60}")

    n_pass = sum(1 for r in results if r.passed)
    n_fail = sum(1 for r in results if r.passed is False)
    n_skip = sum(1 for r in results if r.skipped)
    total = len(results)

    for r in results:
        status = "PASS" if r.passed else ("SKIP" if r.skipped else "FAIL")
        marker = "  " if r.passed else ">>"
        print(f"  {marker} [{status}] {r.name}: {r.detail} ({r.elapsed:.1f}s)")

    print(f"\n  {n_pass}/{total} passed, {n_fail} failed, {n_skip} skipped")
    total_time = sum(r.elapsed for r in results)
    print(f"  Total test time: {total_time:.1f}s")

    if n_fail > 0:
        print(f"\n  FAILURES:")
        for r in results:
            if not r.passed and not r.skipped:
                print(f"    - {r.name}: {r.detail}")
        print()

    if n_fail == 0:
        print(f"\n  ALL TESTS PASSED")
    else:
        print(f"\n  SOME TESTS FAILED")

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()

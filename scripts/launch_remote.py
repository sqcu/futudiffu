"""Remote H100 launch pipeline: 8-phase deterministic validation launcher.

Bootstraps a spot H100 (or local RTX 4090) from zero to validated multi-GPU
training pipeline. Each phase has clear pass/fail semantics and timing.

Usage:
    python scripts/launch_remote.py --model-dir ./models --n-gpus 1
    python scripts/launch_remote.py --model-dir ./models --n-gpus 2 --quick

See docs/remote_deployment.md for full documentation.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"

# Phase result types
PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"
SKIP = "SKIP"


# ---------------------------------------------------------------------------
# Graceful interruption
# ---------------------------------------------------------------------------

_interrupted = False


def _signal_handler(signum, frame):
    global _interrupted
    if _interrupted:
        print("\nForce quit.", flush=True)
        sys.exit(1)
    _interrupted = True
    print("\nInterrupt received. Skipping remaining phases, running summary...",
          flush=True)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _banner(text: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}\n", flush=True)


def _wait_for_port(port: int, timeout: float = 180.0) -> bool:
    """Poll until a TCP port is accepting connections."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _interrupted:
            return False
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return True
        except OSError:
            time.sleep(1)
    return False


def _run_subprocess(
    cmd: list[str],
    env: dict | None = None,
    timeout: float = 600.0,
    label: str = "",
) -> tuple[int, str, str]:
    """Run a subprocess, return (returncode, stdout, stderr)."""
    merged_env = {**os.environ, **(env or {})}
    # Ensure PYTHONPATH includes src/ for imports
    existing_pp = merged_env.get("PYTHONPATH", "")
    if str(SRC_DIR) not in existing_pp:
        merged_env["PYTHONPATH"] = f"{SRC_DIR}{os.pathsep}{existing_pp}" if existing_pp else str(SRC_DIR)

    try:
        proc = subprocess.run(
            cmd,
            env=merged_env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"TIMEOUT after {timeout}s"
    except Exception as e:
        return -1, "", str(e)


def _find_python() -> str:
    """Find the Python interpreter to use.

    Prefers the venv python if it exists (Windows venv from WSL2),
    otherwise falls back to sys.executable.
    """
    # Windows venv path (WSL2 cross-exec)
    venv_py = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_py.exists():
        return str(venv_py)

    # Linux venv
    venv_py_linux = REPO_ROOT / ".venv" / "bin" / "python"
    if venv_py_linux.exists():
        return str(venv_py_linux)

    return sys.executable


# ---------------------------------------------------------------------------
# Server cluster context manager
# ---------------------------------------------------------------------------

class ServerCluster:
    """Launch and manage N inference server subprocesses.

    Usage:
        with ServerCluster(n_gpus=2, base_port=5555, ...) as cluster:
            cluster.wait_ready(timeout=180)
            # use cluster.ports
    """

    def __init__(
        self,
        n_gpus: int,
        base_port: int,
        fp8_diff: str,
        te: str,
        vae: str,
        python: str,
        output_dir: Path,
    ):
        self.n_gpus = n_gpus
        self.base_port = base_port
        self.fp8_diff = fp8_diff
        self.te = te
        self.vae = vae
        self.python = python
        self.output_dir = output_dir
        self.ports = [base_port + i for i in range(n_gpus)]
        self._procs: list[subprocess.Popen] = []
        self._logs: list = []

    def __enter__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        env_base = {
            **os.environ,
            "PYTHONPATH": str(SRC_DIR),
        }

        for i in range(self.n_gpus):
            port = self.ports[i]
            log_path = self.output_dir / f"server_gpu{i}.log"
            log_file = open(log_path, "w")
            self._logs.append(log_file)

            env = {**env_base, "CUDA_VISIBLE_DEVICES": str(i)}

            cmd = [
                self.python, "-m", "futudiffu.server",
                "--port", str(port),
                "--fp8-diff", self.fp8_diff,
                "--te", self.te,
                "--vae", self.vae,
            ]

            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            self._procs.append(proc)
            print(f"  GPU {i}: PID {proc.pid}, port {port}, log {log_path}")

        return self

    def wait_ready(self, timeout: float = 180.0) -> list[bool]:
        """Wait for all servers to become responsive. Returns per-GPU status."""
        results = []
        for i, port in enumerate(self.ports):
            if _interrupted:
                results.append(False)
                continue
            print(f"  Waiting for GPU {i} (port {port})...", end="", flush=True)
            ok = _wait_for_port(port, timeout=timeout)
            # Check if process died
            if self._procs[i].poll() is not None:
                print(f" DEAD (exit code {self._procs[i].returncode})")
                ok = False
            elif ok:
                print(" ready")
            else:
                print(" TIMEOUT")
            results.append(ok)
        return results

    def __exit__(self, *exc):
        print("  Shutting down servers...")
        for i, proc in enumerate(self._procs):
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        # Wait up to 10s for graceful shutdown
        for proc in self._procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        for log_file in self._logs:
            log_file.close()
        print("  All servers stopped.")


# ---------------------------------------------------------------------------
# Phase implementations
# ---------------------------------------------------------------------------

def phase_0_env_check() -> tuple[str, dict]:
    """Phase 0: Environment check."""
    _banner("Phase 0: Environment Check")
    info = {}

    # Python version
    info["python_version"] = sys.version.split()[0]
    print(f"  Python: {info['python_version']}")

    # CUDA availability
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["gpu_count"] = torch.cuda.device_count()
            info["gpus"] = []
            for i in range(info["gpu_count"]):
                props = torch.cuda.get_device_properties(i)
                gpu_info = {
                    "name": props.name,
                    "sm": f"{props.major}.{props.minor}",
                    "vram_gb": round(props.total_memory / (1024**3), 1),
                }
                info["gpus"].append(gpu_info)
                print(f"  GPU {i}: {gpu_info['name']} (SM {gpu_info['sm']}, "
                      f"{gpu_info['vram_gb']} GB)")
        else:
            print("  FAIL: CUDA not available")
            return FAIL, info
    except ImportError:
        print("  FAIL: torch not installed")
        return FAIL, info

    print(f"  torch: {info['torch_version']}")

    # Import checks
    missing = []
    for mod in ["safetensors", "zmq", "triton", "pyarrow"]:
        try:
            __import__(mod)
            print(f"  {mod}: OK")
        except ImportError:
            missing.append(mod)
            print(f"  {mod}: MISSING")

    if missing:
        print(f"  FAIL: Missing packages: {', '.join(missing)}")
        return FAIL, info

    # HF token
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        supersekrit = REPO_ROOT / ".supersekrit"
        if supersekrit.exists():
            for _line in supersekrit.read_text().splitlines():
                _line = _line.strip()
                if _line and not _line.startswith("#"):
                    hf_token = _line
                    break
            info["hf_token_source"] = ".supersekrit"
            print(f"  HF token: found in .supersekrit")
    else:
        info["hf_token_source"] = "HF_TOKEN env"
        print(f"  HF token: found in HF_TOKEN env var")

    if not hf_token:
        # Try huggingface-cli
        try:
            r = subprocess.run(
                ["huggingface-cli", "whoami"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0:
                info["hf_token_source"] = "huggingface-cli"
                print(f"  HF token: found via huggingface-cli ({r.stdout.strip()})")
                hf_token = "cli"
            else:
                print(f"  WARN: No HF token found (downloads may fail for private repos)")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print(f"  WARN: No HF token found")

    info["has_hf_token"] = bool(hf_token)

    return PASS, info


# Canonical model filenames. These are the names remote_node_bootstrap.py
# downloads to, and the names every script in this repo expects. If your
# models live under different names, symlink or copy them -- don't teach
# the loader to guess.
MODEL_FILES = {
    "fp8_diff": "z_image_fp8_blockwise.safetensors",
    "te": "qwen_3_4b.safetensors",
    "vae": "ae.safetensors",
    "bf16_diff": "z_image_bf16.safetensors",  # source for quantization only
}


def phase_1_model_bootstrap(
    model_dir: Path,
    skip_download: bool,
    skip_quantize: bool,
    explicit_fp8: str | None = None,
    explicit_te: str | None = None,
    explicit_vae: str | None = None,
) -> tuple[str, dict]:
    """Phase 1: Download and quantize models.

    Model identification is explicit paths, not runtime discovery.
    --fp8-diff / --te / --vae override --model-dir completely.
    --model-dir is a flat directory with canonical filenames from MODEL_FILES.
    """
    _banner("Phase 1: Model Bootstrap")

    # Resolve the three model paths: explicit flags > model_dir/canonical_name
    fp8_path = Path(explicit_fp8) if explicit_fp8 else model_dir / MODEL_FILES["fp8_diff"]
    te_path = Path(explicit_te) if explicit_te else model_dir / MODEL_FILES["te"]
    vae_path = Path(explicit_vae) if explicit_vae else model_dir / MODEL_FILES["vae"]
    bf16_path = model_dir / MODEL_FILES["bf16_diff"]

    info = {
        "model_dir": str(model_dir),
        "fp8_diff": str(fp8_path),
        "te": str(te_path),
        "vae": str(vae_path),
    }

    # If all three explicit paths exist, skip download/quantize entirely
    if explicit_fp8 and explicit_te and explicit_vae:
        for label, path in [("FP8 diff", fp8_path), ("TE", te_path), ("VAE", vae_path)]:
            if not path.exists():
                print(f"  FAIL: {label} not found at {path}")
                return FAIL, info
            size_gb = path.stat().st_size / (1024**3)
            print(f"  {label}: {path} ({size_gb:.2f} GB)")
        return PASS, info

    model_dir = model_dir.resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    # Import bootstrap functions
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from remote_node_bootstrap import download_models, quantize_diffusion_model

    # Download
    if skip_download:
        print("  Download: skipped (--skip-download)")
    else:
        print("  Downloading models from HuggingFace...")
        try:
            paths = download_models(model_dir)
            info["downloaded"] = {k: str(v) for k, v in paths.items()}
        except Exception as e:
            print(f"  FAIL: Download failed: {e}")
            return FAIL, info

    # Check TE + VAE exist
    if not te_path.exists():
        print(f"  FAIL: {te_path} not found")
        return FAIL, info
    if not vae_path.exists():
        print(f"  FAIL: {vae_path} not found")
        return FAIL, info

    # Quantize BF16 -> FP8 if needed
    if skip_quantize:
        print("  Quantize: skipped (--skip-quantize)")
    elif fp8_path.exists():
        size_gb = fp8_path.stat().st_size / (1024**3)
        print(f"  FP8 model already exists ({size_gb:.2f} GB)")
    else:
        if not bf16_path.exists():
            print(f"  FAIL: {bf16_path} not found (needed for quantization)")
            return FAIL, info
        print("  Quantizing BF16 -> FP8 blockwise...")
        try:
            quantize_diffusion_model(bf16_path, fp8_path, block_size=128)
        except Exception as e:
            print(f"  FAIL: Quantization failed: {e}")
            return FAIL, info

    if not fp8_path.exists():
        print(f"  FAIL: {fp8_path} not found")
        return FAIL, info

    for label, path in [("FP8 diff", fp8_path), ("TE", te_path), ("VAE", vae_path)]:
        size_gb = path.stat().st_size / (1024**3)
        print(f"  {label}: {path} ({size_gb:.2f} GB)")

    return PASS, info


def phase_2_kernel_smoke(
    model_dir: Path,
    python: str,
) -> tuple[str, dict]:
    """Phase 2: Kernel smoke tests (S-S-S model, SageAttention, split piref)."""
    _banner("Phase 2: Kernel Smoke Tests")
    info = {"tests_run": [], "tests_failed": []}

    fp8_path = model_dir / "z_image_fp8_blockwise.safetensors"
    env = {"FUTUDIFFU_FP8_PATH": str(fp8_path)}

    tests = [
        ("test_sage_block_mask", [python, str(REPO_ROOT / "tests" / "test_sage_block_mask.py")]),
        ("test_split_piref", [python, str(REPO_ROOT / "tests" / "test_split_piref.py"), "--iterations", "5"]),
    ]

    for test_name, cmd in tests:
        if _interrupted:
            break
        print(f"  Running {test_name}...", end="", flush=True)
        t0 = time.monotonic()
        rc, stdout, stderr = _run_subprocess(cmd, env=env, timeout=300.0)
        dt = time.monotonic() - t0
        info["tests_run"].append(test_name)

        if rc == 0:
            print(f" PASS ({dt:.1f}s)")
        else:
            print(f" FAIL ({dt:.1f}s)")
            info["tests_failed"].append(test_name)
            # Print last 20 lines of output for diagnosis
            output = (stderr or stdout).strip().splitlines()
            for line in output[-20:]:
                print(f"    {line}")

    if info["tests_failed"]:
        print(f"\n  WARN: {len(info['tests_failed'])} kernel test(s) failed")
        print(f"  (Phase 4 pipeline validation is the true correctness gate)")
        return WARN, info

    print(f"\n  All kernel tests passed")
    return PASS, info


def phase_3_server_launch(
    n_gpus: int,
    base_port: int,
    model_info: dict,
    python: str,
    output_dir: Path,
) -> tuple[str, dict, ServerCluster | None]:
    """Phase 3: Launch server cluster.

    Returns (status, info, cluster). Caller must manage cluster lifetime.
    """
    _banner("Phase 3: Server Launch")
    info = {"n_gpus": n_gpus, "ports": list(range(base_port, base_port + n_gpus))}

    cluster = ServerCluster(
        n_gpus=n_gpus,
        base_port=base_port,
        fp8_diff=model_info["fp8_diff"],
        te=model_info["te"],
        vae=model_info["vae"],
        python=python,
        output_dir=output_dir,
    )
    cluster.__enter__()

    print()
    ready = cluster.wait_ready(timeout=180.0)
    info["ready"] = ready

    if not all(ready):
        failed = [i for i, ok in enumerate(ready) if not ok]
        print(f"\n  FAIL: GPUs {failed} did not become ready")
        cluster.__exit__(None, None, None)
        return FAIL, info, None

    print(f"\n  All {n_gpus} server(s) ready")
    return PASS, info, cluster


def phase_4_pipeline_validation(
    port: int,
    output_dir: Path,
) -> tuple[str, dict]:
    """Phase 4: Pipeline validation against reference trajectory."""
    _banner("Phase 4: Pipeline Validation")

    ref_dir = REPO_ROOT / "stream_futudiffu"
    val_dir = output_dir / "pipeline"

    if not ref_dir.exists():
        print(f"  WARN: Reference directory {ref_dir} not found, skipping")
        return WARN, {"reason": "no_reference_dir"}

    # Import and run the validation
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        from validate_server_pipeline import validate_pipeline
        passed, stats = validate_pipeline(
            port=port,
            ref_dir=str(ref_dir),
            output_dir=str(val_dir),
        )
    except Exception as e:
        print(f"\n  FAIL: Validation crashed: {e}")
        import traceback
        traceback.print_exc()
        return FAIL, {"error": str(e)}

    # Interpret results with cross-arch thresholds
    final_cos = stats.get("final_cos", 0)
    vae_cos = stats.get("vae_cos", 0)

    if final_cos < 0.90 or vae_cos < 0.90:
        print(f"\n  FAIL: Below minimum threshold (cos < 0.90)")
        return FAIL, stats

    if final_cos < 0.99 or vae_cos < 0.99:
        print(f"\n  WARN: Cross-architecture divergence detected (0.90 <= cos < 0.99)")
        print(f"  This is EXPECTED when running on a different GPU architecture (e.g. SM89 -> SM90)")
        return WARN, stats

    print(f"\n  PASS: Same-architecture match (cos >= 0.99)")
    return PASS, stats


def phase_5_gen_stub(
    ports: list[int],
    python: str,
    output_dir: Path,
) -> tuple[str, dict]:
    """Phase 5: Dataset generation stub (2 trajectories per GPU)."""
    _banner("Phase 5: Dataset Generation Stub")
    info = {"n_gpus": len(ports)}

    gen_script = REPO_ROOT / "scripts" / "generate_btrm_dataset.py"
    staging_dirs = []
    procs = []

    # Launch one generation per GPU in parallel
    for i, port in enumerate(ports):
        if _interrupted:
            break
        staging_dir = output_dir / f"gen_stub_gpu{i}"
        staging_dirs.append(staging_dir)

        cmd = [
            python, str(gen_script),
            "--t2i", "2",
            "--server", f"tcp://localhost:{port}",
            "--output-dir", str(staging_dir),
            "--gpu-id", str(i),
            "--render", "1",
            "--dataset-format", "v2",
        ]
        print(f"  GPU {i}: generating 2 t2i trajectories -> {staging_dir}")
        env = {"PYTHONPATH": str(SRC_DIR)}
        merged_env = {**os.environ, **env}
        proc = subprocess.Popen(
            cmd, env=merged_env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        procs.append((i, proc))

    # Wait for all
    results = []
    for i, proc in procs:
        try:
            stdout, _ = proc.communicate(timeout=300)
            results.append((i, proc.returncode, stdout))
        except subprocess.TimeoutExpired:
            proc.kill()
            results.append((i, -1, "TIMEOUT"))

    failed = [(i, rc) for i, rc, _ in results if rc != 0]
    if failed:
        for i, rc, out in results:
            if rc != 0:
                print(f"  GPU {i}: FAIL (rc={rc})")
                for line in (out or "").strip().splitlines()[-10:]:
                    print(f"    {line}")
        info["failed_gpus"] = [i for i, _ in failed]
        return WARN, info

    # Merge if multi-GPU
    if len(staging_dirs) > 1:
        merge_script = REPO_ROOT / "scripts" / "merge_staged_datasets.py"
        merged_dir = output_dir / "gen_stub_merged"
        actual_dirs = [str(d) + f"_gpu{i}" for i, d in enumerate(staging_dirs)]
        cmd = [
            python, str(merge_script),
            "--staging-dirs", *actual_dirs,
            "--output", str(merged_dir),
        ]
        print(f"\n  Merging {len(actual_dirs)} staging dirs -> {merged_dir}")
        rc, stdout, stderr = _run_subprocess(cmd, timeout=120)
        if rc != 0:
            print(f"  WARN: Merge failed (rc={rc})")
            print(f"  {stderr[-500:] if stderr else stdout[-500:]}")
            return WARN, info
        print(f"  Merge complete")

    info["staging_dirs"] = [str(d) for d in staging_dirs]
    print(f"\n  Generation stub complete: {len(ports)} GPUs, 2 trajectories each")
    return PASS, info


def phase_6_btrm_stub(
    ports: list[int],
    python: str,
    dataset_dir: Path,
    output_dir: Path,
) -> tuple[str, dict]:
    """Phase 6: BTRM training stub (3 macrobatches)."""
    _banner("Phase 6: BTRM Training Stub")
    info = {}

    train_script = REPO_ROOT / "scripts" / "train.py"
    btrm_output = output_dir / "btrm_stub"

    port_args = []
    for p in ports:
        port_args.extend(["--ports", str(p)])
    # Flatten: --ports p1 p2 p3
    port_args = ["--ports"] + [str(p) for p in ports]

    cmd = [
        python, str(train_script),
        *port_args,
        "--dataset-dir", str(dataset_dir),
        "--output-dir", str(btrm_output),
        "--btrm-macrobatches", "3",
        "--btrm-batch-size", "16",
        "--skip-policy",
        "--checkpoint-every", "0",
        "--render-every", "0",
    ]

    print(f"  Running: {' '.join(cmd[-10:])}")
    print(f"  Dataset: {dataset_dir}")
    print(f"  Output: {btrm_output}")

    rc, stdout, stderr = _run_subprocess(cmd, timeout=600)
    output_text = stderr or stdout

    if rc != 0:
        print(f"\n  WARN: BTRM stub failed (rc={rc})")
        for line in output_text.strip().splitlines()[-20:]:
            print(f"    {line}")
        info["returncode"] = rc
        return WARN, info

    # Parse metrics.jsonl for validation
    metrics_path = btrm_output / "metrics.jsonl"
    if metrics_path.exists():
        lines = metrics_path.read_text().strip().splitlines()
        btrm_steps = []
        for line in lines:
            try:
                record = json.loads(line)
                if record.get("phase") == "btrm":
                    btrm_steps.append(record)
            except json.JSONDecodeError:
                pass

        if btrm_steps:
            losses = [s["loss"] for s in btrm_steps]
            info["n_steps"] = len(btrm_steps)
            info["losses"] = losses
            info["loss_is_finite"] = all(
                not (isinstance(l, float) and (l != l or abs(l) == float("inf")))
                for l in losses
            )

            print(f"\n  BTRM steps: {len(btrm_steps)}")
            for s in btrm_steps:
                acc = s.get("per_head_accuracy", {})
                acc_str = ", ".join(f"{k}={v:.2%}" for k, v in acc.items())
                print(f"    step {s['step']}: loss={s['loss']:.4f}, acc=[{acc_str}]")

            if not info["loss_is_finite"]:
                print(f"\n  WARN: Non-finite loss detected")
                return WARN, info
        else:
            print(f"\n  WARN: No BTRM training steps found in metrics")
            return WARN, info
    else:
        print(f"\n  WARN: No metrics.jsonl found at {metrics_path}")
        return WARN, info

    print(f"\n  BTRM stub complete")
    return PASS, info


def phase_7_policy_stub(
    ports: list[int],
    python: str,
    dataset_dir: Path,
    output_dir: Path,
) -> tuple[str, dict]:
    """Phase 7: Policy training stub (3 iterations, group_size=2)."""
    _banner("Phase 7: Policy Training Stub")
    info = {}

    train_script = REPO_ROOT / "scripts" / "train.py"
    policy_output = output_dir / "policy_stub"

    port_args = ["--ports"] + [str(p) for p in ports]

    cmd = [
        python, str(train_script),
        *port_args,
        "--dataset-dir", str(dataset_dir),
        "--output-dir", str(policy_output),
        "--skip-btrm",
        "--resume",
        "--policy-iterations", "3",
        "--policy-group-size", "2",
        "--policy-rollout-steps", "10",
        "--checkpoint-every", "0",
        "--render-every", "0",
    ]

    print(f"  Running policy stub (3 iterations, group_size=2)...")
    print(f"  Output: {policy_output}")

    rc, stdout, stderr = _run_subprocess(cmd, timeout=600)
    output_text = stderr or stdout

    if rc != 0:
        print(f"\n  WARN: Policy stub failed (rc={rc})")
        for line in output_text.strip().splitlines()[-20:]:
            print(f"    {line}")
        info["returncode"] = rc
        return WARN, info

    # Parse metrics.jsonl
    metrics_path = policy_output / "metrics.jsonl"
    if metrics_path.exists():
        lines = metrics_path.read_text().strip().splitlines()
        policy_steps = []
        for line in lines:
            try:
                record = json.loads(line)
                if record.get("phase") == "policy":
                    policy_steps.append(record)
            except json.JSONDecodeError:
                pass

        if policy_steps:
            info["n_iters"] = len(policy_steps)
            grad_norms = [s.get("grad_norm", 0) for s in policy_steps]
            info["grad_norms"] = grad_norms
            info["has_nonzero_grad"] = any(g > 0 for g in grad_norms)

            print(f"\n  Policy iterations: {len(policy_steps)}")
            for s in policy_steps:
                rewards = s.get("rewards", [])
                rew_str = ", ".join(f"{r:+.4f}" for r in rewards)
                print(f"    iter {s['iter']}: rewards=[{rew_str}], "
                      f"grad_norm={s.get('grad_norm', 0):.3e}")

            if not info["has_nonzero_grad"]:
                print(f"\n  WARN: All grad_norms are zero")
                return WARN, info
        else:
            print(f"\n  WARN: No policy iterations found in metrics")
            return WARN, info
    else:
        print(f"\n  WARN: No metrics.jsonl found")
        return WARN, info

    print(f"\n  Policy stub complete")
    return PASS, info


# ---------------------------------------------------------------------------
# Phase 8: Summary (always runs)
# ---------------------------------------------------------------------------

def phase_8_summary(
    results: dict[str, tuple[str, float, dict]],
    args: argparse.Namespace,
    model_info: dict | None,
    output_dir: Path,
) -> None:
    """Phase 8: Print summary table and save launch report."""
    _banner("Phase 8: Launch Summary")

    # Timing table
    print(f"  {'Phase':<30} {'Status':<8} {'Time':>8}")
    print(f"  {'-' * 50}")
    total_time = 0.0
    for phase_name, (status, elapsed, _info) in results.items():
        total_time += elapsed
        time_str = f"{elapsed:.1f}s" if elapsed > 0 else "-"
        print(f"  {phase_name:<30} {status:<8} {time_str:>8}")
    print(f"  {'-' * 50}")
    print(f"  {'TOTAL':<30} {'':8} {total_time:>7.1f}s")

    # Overall status
    statuses = [s for s, _, _ in results.values()]
    if FAIL in statuses:
        overall = "FAILED"
    elif WARN in statuses:
        overall = "PASSED WITH WARNINGS"
    else:
        overall = "ALL PASSED"
    print(f"\n  Overall: {overall}")

    # Suggested real-run command
    if model_info and overall != "FAILED":
        ports = results.get("3_server_launch", (None, 0, {}))[2].get("ports", [args.base_port])
        port_str = " ".join(str(p) for p in ports)

        print(f"\n  Suggested real training run:")
        print(f"    python scripts/train.py \\")
        print(f"      --ports {port_str} \\")
        print(f"      --dataset-dir {args.dataset_dir} \\")
        print(f"      --output-dir ./training_output \\")
        print(f"      --btrm-macrobatches 30 \\")
        print(f"      --policy-iterations 50")

    # Save report
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "args": vars(args),
        "overall": overall,
        "total_time_s": round(total_time, 1),
        "phases": {
            name: {"status": status, "elapsed_s": round(elapsed, 1), "info": info}
            for name, (status, elapsed, info) in results.items()
        },
    }

    report_path = output_dir / "launch_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  Report saved: {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remote H100 launch pipeline: 8-phase deterministic validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--model-dir", type=str, default="./models",
                        help="Where to download/find models")
    parser.add_argument("--fp8-diff", type=str, default=None,
                        help="Explicit path to FP8 diffusion model (overrides --model-dir)")
    parser.add_argument("--te", type=str, default=None,
                        help="Explicit path to text encoder (overrides --model-dir)")
    parser.add_argument("--vae", type=str, default=None,
                        help="Explicit path to VAE (overrides --model-dir)")
    parser.add_argument("--n-gpus", type=int, default=0,
                        help="Number of GPUs (0 = auto-detect)")
    parser.add_argument("--base-port", type=int, default=5555,
                        help="Starting port for servers")

    # Skip flags
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip HF model download")
    parser.add_argument("--skip-quantize", action="store_true",
                        help="Skip BF16 -> FP8 quantization")
    parser.add_argument("--skip-kernel-test", action="store_true",
                        help="Skip S-S-S kernel smoke tests (Phase 2)")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip reference trajectory comparison (Phase 4)")
    parser.add_argument("--skip-gen-stub", action="store_true",
                        help="Skip dataset generation stub (Phase 5)")
    parser.add_argument("--skip-btrm-stub", action="store_true",
                        help="Skip BTRM training stub (Phase 6)")
    parser.add_argument("--skip-policy-stub", action="store_true",
                        help="Skip policy training stub (Phase 7)")

    # Paths
    parser.add_argument("--dataset-dir", type=str, default="btrm_dataset",
                        help="BTRM dataset directory")
    parser.add_argument("--output-dir", type=str, default="remote_validation",
                        help="Validation output directory")

    # Convenience aliases
    parser.add_argument("--quick", action="store_true",
                        help="Alias: --skip-kernel-test --skip-gen-stub")

    args = parser.parse_args()

    # Apply aliases
    if args.quick:
        args.skip_kernel_test = True
        args.skip_gen_stub = True

    # Resolve paths
    model_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()

    # Auto-detect GPU count
    if args.n_gpus == 0:
        try:
            import torch
            args.n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except ImportError:
            args.n_gpus = 0

    # Install signal handler
    signal.signal(signal.SIGINT, _signal_handler)

    python = _find_python()
    results: dict[str, tuple[str, float, dict]] = {}
    model_info: dict | None = None
    cluster: ServerCluster | None = None
    summary_printed = False

    _banner(f"futudiffu Remote Launch Pipeline")
    print(f"  Model dir:   {model_dir}")
    print(f"  Output dir:  {output_dir}")
    print(f"  Dataset dir: {dataset_dir}")
    print(f"  GPUs:        {args.n_gpus}")
    print(f"  Base port:   {args.base_port}")
    print(f"  Python:      {python}")
    print(f"  Quick mode:  {args.quick}")

    try:
        # Phase 0: Environment Check -- FAIL = abort
        if not _interrupted:
            t0 = time.monotonic()
            status, info = phase_0_env_check()
            results["0_env_check"] = (status, time.monotonic() - t0, info)
            if status == FAIL:
                phase_8_summary(results, args, model_info, output_dir)
                summary_printed = True
                return 1

            # Update n_gpus from detection if needed
            if args.n_gpus == 0 and "gpu_count" in info:
                args.n_gpus = info["gpu_count"]

        # Phase 1: Model Bootstrap -- FAIL = abort
        if not _interrupted:
            t0 = time.monotonic()
            status, info = phase_1_model_bootstrap(
                model_dir, args.skip_download, args.skip_quantize,
                explicit_fp8=args.fp8_diff,
                explicit_te=args.te,
                explicit_vae=args.vae,
            )
            results["1_model_bootstrap"] = (status, time.monotonic() - t0, info)
            if status == FAIL:
                phase_8_summary(results, args, model_info, output_dir)
                summary_printed = True
                return 1
            model_info = info

        # Phase 2: Kernel Smoke -- FAIL = warn + continue
        if not _interrupted and not args.skip_kernel_test:
            t0 = time.monotonic()
            status, info = phase_2_kernel_smoke(model_dir, python)
            results["2_kernel_smoke"] = (status, time.monotonic() - t0, info)
        elif not _interrupted:
            results["2_kernel_smoke"] = (SKIP, 0, {"reason": "skipped"})

        # Phase 3: Server Launch -- FAIL = abort
        if not _interrupted and model_info:
            t0 = time.monotonic()
            status, info, cluster = phase_3_server_launch(
                args.n_gpus, args.base_port, model_info, python, output_dir)
            results["3_server_launch"] = (status, time.monotonic() - t0, info)
            if status == FAIL:
                phase_8_summary(results, args, model_info, output_dir)
                summary_printed = True
                return 1

        # Phase 4: Pipeline Validation -- FAIL = abort
        if not _interrupted and cluster and not args.skip_validation:
            t0 = time.monotonic()
            status, info = phase_4_pipeline_validation(
                args.base_port, output_dir)
            results["4_pipeline_validation"] = (status, time.monotonic() - t0, info)
            if status == FAIL:
                phase_8_summary(results, args, model_info, output_dir)
                summary_printed = True
                return 1
        elif not _interrupted:
            results["4_pipeline_validation"] = (SKIP, 0, {"reason": "skipped"})

        # Phase 5: Generation Stub -- FAIL = warn + continue
        if not _interrupted and cluster and not args.skip_gen_stub:
            t0 = time.monotonic()
            ports = cluster.ports if cluster else [args.base_port]
            status, info = phase_5_gen_stub(ports, python, output_dir)
            results["5_gen_stub"] = (status, time.monotonic() - t0, info)
        elif not _interrupted:
            results["5_gen_stub"] = (SKIP, 0, {"reason": "skipped"})

        # Phase 6: BTRM Stub -- FAIL = warn + continue
        if not _interrupted and cluster and not args.skip_btrm_stub:
            t0 = time.monotonic()
            ports = cluster.ports if cluster else [args.base_port]
            status, info = phase_6_btrm_stub(
                ports, python, dataset_dir, output_dir)
            results["6_btrm_stub"] = (status, time.monotonic() - t0, info)
        elif not _interrupted:
            results["6_btrm_stub"] = (SKIP, 0, {"reason": "skipped"})

        # Phase 7: Policy Stub -- FAIL = warn + continue
        if not _interrupted and cluster and not args.skip_policy_stub:
            t0 = time.monotonic()
            ports = cluster.ports if cluster else [args.base_port]
            status, info = phase_7_policy_stub(
                ports, python, dataset_dir, output_dir)
            results["7_policy_stub"] = (status, time.monotonic() - t0, info)
        elif not _interrupted:
            results["7_policy_stub"] = (SKIP, 0, {"reason": "skipped"})

    finally:
        # Phase 8: Summary (always runs, but only once)
        if not summary_printed:
            phase_8_summary(results, args, model_info, output_dir)

        # Cleanup servers
        if cluster is not None:
            cluster.__exit__(None, None, None)

    # Return code
    statuses = [s for s, _, _ in results.values()]
    if FAIL in statuses:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

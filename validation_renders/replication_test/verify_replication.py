"""
Verification script for the replication test.

Checks that all required artifacts exist, are valid, and cross-run
comparisons produce expected results.

Run AFTER the replication test completes:
    .venv\\Scripts\\python.exe F:\\dox\\repos\\ai\\futudiffu\\validation_renders\\replication_test\\verify_replication.py
"""

import sys
import os
import json
import datetime

sys.path.insert(0, r"F:\dox\repos\ai\futudiffu\src")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Base directory: resolve relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = SCRIPT_DIR  # validation_renders/replication_test/

RUN_DIRS = ["run_a", "run_b"]
STEP_FILES = [f"step_{i:02d}.pt" for i in range(30)] + ["final.pt"]

IMAGE_GROUPS = {
    "run_a_render": ["run_a_render.ppm", "run_a_render.png"],
    "run_b_render": ["run_b_render.ppm", "run_b_render.png"],
    "diff_10x": ["diff_10x.ppm", "diff_10x.png"],
}

EXPECTED_SHAPE = (1, 16, 104, 160)

REPORT_PATH = os.path.join(BASE_DIR, "verification_report.txt")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class CheckResult:
    def __init__(self, name: str, passed: bool, detail: str = ""):
        self.name = name
        self.passed = passed
        self.detail = detail

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        line = f"[{status}] {self.name}"
        if self.detail:
            line += f"  --  {self.detail}"
        return line


def file_size_str(path: str) -> str:
    """Human-readable file size."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return "N/A"
    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    else:
        return f"{size / (1024 * 1024):.2f} MB"


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

results: list[CheckResult] = []
file_inventory: list[str] = []


def record(name: str, passed: bool, detail: str = ""):
    r = CheckResult(name, passed, detail)
    results.append(r)
    return r


def inventory(path: str, label: str):
    exists = os.path.isfile(path)
    size = file_size_str(path) if exists else "MISSING"
    file_inventory.append(f"{'EXISTS' if exists else 'MISSING':>7}  {size:>12}  {label}")
    return exists


# ---- 1. Directory structure ----

for run in RUN_DIRS:
    run_dir = os.path.join(BASE_DIR, run)
    dir_exists = os.path.isdir(run_dir)
    record(f"Directory {run}/ exists", dir_exists)

    for fname in STEP_FILES:
        fpath = os.path.join(run_dir, fname)
        exists = inventory(fpath, f"{run}/{fname}")
        record(f"File {run}/{fname} exists", exists)


# ---- 2. Image files ----

for group_name, candidates in IMAGE_GROUPS.items():
    found_any = False
    for cand in candidates:
        cpath = os.path.join(BASE_DIR, cand)
        exists = inventory(cpath, cand)
        if exists:
            found_any = True
    record(
        f"Image group '{group_name}' has at least one file",
        found_any,
        detail=f"checked: {', '.join(candidates)}",
    )


# ---- 3. Analysis output ----

stats_path = os.path.join(BASE_DIR, "residual_analysis", "stats.json")
summary_path = os.path.join(BASE_DIR, "residual_analysis", "summary.txt")

stats_exists = inventory(stats_path, "residual_analysis/stats.json")
summary_exists = inventory(summary_path, "residual_analysis/summary.txt")

# stats.json: valid JSON with "per_step" key
stats_valid = False
stats_has_per_step = False
if stats_exists:
    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            stats_data = json.load(f)
        stats_valid = True
        stats_has_per_step = "per_step" in stats_data
    except (json.JSONDecodeError, OSError) as e:
        stats_valid = False

record("residual_analysis/stats.json exists", stats_exists)
record("residual_analysis/stats.json is valid JSON", stats_valid)
record(
    "residual_analysis/stats.json has 'per_step' key",
    stats_has_per_step,
)

# summary.txt: non-empty, contains "OVERALL CHARACTERIZATION"
summary_valid = False
summary_has_keyword = False
if summary_exists:
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_text = f.read()
        summary_valid = len(summary_text.strip()) > 0
        summary_has_keyword = "OVERALL CHARACTERIZATION" in summary_text
    except OSError:
        summary_valid = False

record("residual_analysis/summary.txt exists", summary_exists)
record("residual_analysis/summary.txt is non-empty", summary_valid)
record(
    "residual_analysis/summary.txt contains 'OVERALL CHARACTERIZATION'",
    summary_has_keyword,
)


# ---- 4. Tensor validity ----

import torch

tensor_load_failures: list[str] = []
tensor_shape_failures: list[str] = []
tensor_dtype_failures: list[str] = []

for run in RUN_DIRS:
    for fname in STEP_FILES:
        fpath = os.path.join(BASE_DIR, run, fname)
        if not os.path.isfile(fpath):
            # Already flagged in check 1; skip tensor checks for missing files
            continue

        # Load
        try:
            t = torch.load(fpath, map_location="cpu", weights_only=True)
        except Exception as e:
            tensor_load_failures.append(f"{run}/{fname}: {e}")
            continue

        # Shape
        if t.shape != EXPECTED_SHAPE:
            tensor_shape_failures.append(
                f"{run}/{fname}: got {tuple(t.shape)}, expected {EXPECTED_SHAPE}"
            )

        # Dtype
        if t.dtype != torch.bfloat16:
            tensor_dtype_failures.append(
                f"{run}/{fname}: got {t.dtype}, expected bfloat16"
            )

record(
    "All .pt files load successfully",
    len(tensor_load_failures) == 0,
    detail="; ".join(tensor_load_failures) if tensor_load_failures else "all loaded",
)
record(
    "All step tensors have shape (1, 16, 104, 160)",
    len(tensor_shape_failures) == 0,
    detail="; ".join(tensor_shape_failures) if tensor_shape_failures else "all correct",
)
record(
    "All step tensors are bfloat16",
    len(tensor_dtype_failures) == 0,
    detail="; ".join(tensor_dtype_failures) if tensor_dtype_failures else "all correct",
)


# ---- 5. Cross-run comparison ----

def compute_mse(path_a: str, path_b: str) -> float | None:
    """Load two tensors and compute MSE. Returns None on failure."""
    try:
        a = torch.load(path_a, map_location="cpu", weights_only=True).float()
        b = torch.load(path_b, map_location="cpu", weights_only=True).float()
        return ((a - b) ** 2).mean().item()
    except Exception:
        return None


comparison_pairs = [
    ("step_00.pt", "Step 00"),
    ("final.pt", "Final latent"),
]

for fname, label in comparison_pairs:
    path_a = os.path.join(BASE_DIR, "run_a", fname)
    path_b = os.path.join(BASE_DIR, "run_b", fname)

    if not (os.path.isfile(path_a) and os.path.isfile(path_b)):
        record(
            f"Cross-run MSE for {label}",
            False,
            detail="one or both files missing",
        )
        continue

    mse = compute_mse(path_a, path_b)
    if mse is None:
        record(f"Cross-run MSE for {label}", False, detail="failed to compute MSE")
        continue

    if mse == 0.0:
        mse_desc = "BITWISE IDENTICAL (MSE = 0.0)"
    elif mse < 1e-6:
        mse_desc = f"Effectively identical (MSE = {mse:.2e} < 1e-6)"
    elif mse < 0.01:
        mse_desc = f"Minor divergence (MSE = {mse:.6e})"
    else:
        mse_desc = f"UNEXPECTED DIVERGENCE (MSE = {mse:.6e} > 0.01)"

    # Cross-run comparison is informational: pass unless unexpected divergence
    is_unexpected = mse > 0.01
    record(
        f"Cross-run MSE for {label}",
        not is_unexpected,
        detail=mse_desc,
    )

# Also compute cosine similarity for final latent
path_a_final = os.path.join(BASE_DIR, "run_a", "final.pt")
path_b_final = os.path.join(BASE_DIR, "run_b", "final.pt")
if os.path.isfile(path_a_final) and os.path.isfile(path_b_final):
    try:
        a = torch.load(path_a_final, map_location="cpu", weights_only=True).float().flatten()
        b = torch.load(path_b_final, map_location="cpu", weights_only=True).float().flatten()
        cos = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
        record(
            "Cross-run cosine similarity (final)",
            cos > 0.99,
            detail=f"cos = {cos:.6f}",
        )
    except Exception as e:
        record("Cross-run cosine similarity (final)", False, detail=str(e))


# ---- 6. Summary report ----

total_checks = len(results)
passed = sum(1 for r in results if r.passed)
failed = total_checks - passed
overall = "PASS" if failed == 0 else "FAIL"

report_lines = []
report_lines.append("=" * 78)
report_lines.append("  REPLICATION TEST VERIFICATION REPORT")
report_lines.append(f"  Generated: {datetime.datetime.now().isoformat()}")
report_lines.append("=" * 78)
report_lines.append("")

report_lines.append("--- FILE INVENTORY ---")
report_lines.append(f"{'STATUS':>7}  {'SIZE':>12}  PATH")
report_lines.append("-" * 78)
for line in file_inventory:
    report_lines.append(line)
report_lines.append("")

report_lines.append("--- CHECK RESULTS ---")
report_lines.append("-" * 78)
for r in results:
    report_lines.append(str(r))
report_lines.append("")

report_lines.append("--- SUMMARY ---")
report_lines.append(f"Total checks: {total_checks}")
report_lines.append(f"Passed:       {passed}")
report_lines.append(f"Failed:       {failed}")
report_lines.append("")
report_lines.append(f"OVERALL: {overall}")
report_lines.append("=" * 78)

report_text = "\n".join(report_lines) + "\n"

# Write report
os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report_text)

# Also print to stdout
print(report_text)

# Exit code
sys.exit(0 if overall == "PASS" else 1)

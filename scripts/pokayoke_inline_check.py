"""pokayoke_inline_check.py -- mechanical boundary enforcement for src_ii/scripts_ii.

Enforces three classes of constraint:

  Check 1 -- Inlining prevention (scripts_ii/ only):
    Scripts must NOT implement rendering pipelines, sigma schedules, or
    finite-difference utilities inline. These belong in src_ii modules.

  Check 2 -- Name collision prevention (scripts_ii/ only):
    Scripts must NOT define functions that share names with public functions
    in src_ii modules (silent duplication risk).

  Check 3 -- src/futudiffu/ freeze enforcement (src_ii/ AND scripts_ii/):
    No file in src_ii/ or scripts_ii/ may import from the frozen
    src/futudiffu/ package. This includes:
      - `from futudiffu.X import Y`
      - `import futudiffu.X`
      - `sys.path.insert(0, .../src)` (path manipulation enabling the above)
    Existing violations are grandfathered in SRC_FREEZE_EXCEPTIONS and must
    shrink monotonically as imports are migrated to src_ii equivalents.

Exit codes:
  0  -- all checks pass (or all violations are grandfathered)
  1  -- at least one ungrandfathered violation found

Grandfathering:
  POKAYOKE_EXCEPTIONS -- (filename, pattern_id) for inlining violations
  NAME_COLLISION_EXCEPTIONS -- (filename, function_name) for name collisions
  SRC_FREEZE_EXCEPTIONS -- (directory, filename, module_or_pattern) for frozen imports

  All exception lists should shrink monotonically. Adding a new entry
  requires a comment explaining why it is temporary.

Running:
  python scripts/pokayoke_inline_check.py
  python -m py_compile scripts/pokayoke_inline_check.py  # syntax check only
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository layout
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_II_DIR = REPO_ROOT / "scripts_ii"
SRC_II_DIR = REPO_ROOT / "src_ii"


# ---------------------------------------------------------------------------
# Inlining pattern definitions
# ---------------------------------------------------------------------------
#
# Each entry is a tuple:
#   (pattern_id, regex, human_readable_message)
#
# pattern_id is used as the key in POKAYOKE_EXCEPTIONS.
# regex is matched against the full file text (not line by line).
#
# On a match, the script reports the file and line number.

FORBIDDEN_PATTERNS: list[tuple[str, str, str]] = [
    # ------------------------------------------------------------------
    # Pattern group 1: Tensor-to-PNG pipeline
    # The canonical 7-step pipeline: squeeze -> clamp -> permute(1,2,0)
    # -> cpu() -> numpy() -> *255 -> uint8 -> PIL.fromarray -> save.
    # Any script implementing this inline should import
    # src_ii.rendering.save_tensor_as_png instead.
    # ------------------------------------------------------------------
    (
        "tensor_to_png_permute",
        r"\.permute\s*\(\s*1\s*,\s*2\s*,\s*0\s*\)\s*\.cpu\s*\(\s*\)\s*\.numpy\s*\(\s*\)",
        "Inlined tensor-to-PNG permute/cpu/numpy pipeline "
        "(use src_ii.rendering.save_tensor_as_png)",
    ),
    (
        "tensor_to_png_astype_uint8",
        r"\.astype\s*\(\s*np\.uint8\s*\)",
        "Inlined astype(np.uint8) conversion in tensor-to-PNG pipeline "
        "(use src_ii.rendering.save_tensor_as_png)",
    ),
    (
        "tensor_to_png_byte_numpy_fromarray",
        r"\.byte\s*\(\s*\)\s*\.numpy\s*\(\s*\)",
        "Inlined .byte().numpy() in tensor-to-image pipeline "
        "(use src_ii.rendering.save_tensor_as_png or make_false_color_diff)",
    ),
    # ------------------------------------------------------------------
    # Pattern group 2: False-color diff inlining
    # Inline abs(a - b) * 10.0 for diff visualization belongs in
    # src_ii.rendering.save_false_color_diff.
    # ------------------------------------------------------------------
    (
        "false_color_diff_abs_scale",
        r"\.abs\s*\(\s*\)\s*\*\s*10(?:\.0)?(?!\d)",
        "Inlined false-color diff scale (.abs() * 10) "
        "(use src_ii.rendering.save_false_color_diff)",
    ),
    # ------------------------------------------------------------------
    # Pattern group 3: Sigma schedule inlining
    # Karras / ComfyUI sigma schedules belong in src_ii.sigma_schedule.
    # ------------------------------------------------------------------
    (
        "sigma_schedule_karras_formula",
        r"sigma_max\s*\*\*\s*\(\s*1\s*/\s*rho\s*\)",
        "Inlined Karras sigma schedule formula "
        "(use src_ii.sigma_schedule.build_sigma_schedule)",
    ),
    (
        "sigma_schedule_function_def",
        r"def\s+_build_sigma_schedule\s*\(",
        "Inlined sigma schedule function definition "
        "(use src_ii.sigma_schedule.build_sigma_schedule)",
    ),
    # ------------------------------------------------------------------
    # Pattern group 4: Finite differences inlining
    # Generic numeric utilities belong in src_ii.stats.
    # ------------------------------------------------------------------
    (
        "finite_diff_function_def",
        r"def\s+(?:finite_diff|compute_finite_diff(?:erences?)?)\s*\(",
        "Inlined finite-differences function definition "
        "(use src_ii.stats.finite_differences)",
    ),
]


# ---------------------------------------------------------------------------
# Grandfathered exceptions
# ---------------------------------------------------------------------------
#
# Format: (filename_stem, pattern_id)
#   filename_stem: just the filename, no directory (e.g. "validate_packed_vs_serial.py")
#   pattern_id: one of the pattern_id strings in FORBIDDEN_PATTERNS
#
# Each entry must include a comment explaining what it tracks and when it
# can be removed (i.e., when the canonical src_ii module is imported instead).
#
# Remove an entry only after the canonical import exists and the inline code
# has been deleted.

POKAYOKE_EXCEPTIONS: list[tuple[str, str]] = [
    # -----------------------------------------------------------------------
    # validate_packed_vs_serial.py -- permute(1,2,0).cpu().numpy() still
    # appears at line ~529 as a preprocessing step for
    # compute_spatial_autocorrelation (not part of the tensor-to-PNG
    # pipeline). The save_tensor_as_png inline has been removed and replaced
    # with an import from src_ii.rendering; .astype(np.uint8) and
    # false_color_diff_abs_scale violations are also gone. Only this
    # single residual pattern remains because compute_spatial_autocorrelation
    # receives a numpy array and the permute/cpu/numpy conversion is the
    # correct way to prepare it.
    # Remove when: compute_spatial_autocorrelation accepts a torch tensor
    # directly, or the call site is refactored to use an intermediate helper.
    # -----------------------------------------------------------------------
    ("validate_packed_vs_serial.py", "tensor_to_png_permute"),
]

# Convert exceptions to a set for O(1) lookup.
_EXCEPTION_SET: set[tuple[str, str]] = set(POKAYOKE_EXCEPTIONS)


# ---------------------------------------------------------------------------
# Name collision check
# ---------------------------------------------------------------------------
#
# Public functions in src_ii modules (current + proposed rendering.py API).
# A function defined in scripts_ii/ that shares a name with one of these
# is a structural violation: it suggests the script is implementing something
# that already belongs (or is proposed to belong) in a canonical module.
#
# Naming convention:
#   - Names prefixed with _ are private and excluded from src_ii's public API.
#   - Only top-level function definitions (def at column 0) are checked in
#     scripts_ii/. Methods inside classes are not flagged.
#
# Proposed rendering.py functions are included because the pokayoke is
# designed to prevent NEW violations, not just document existing ones.
# If rendering.py is created, scripts should import from it rather than
# defining their own copies.

SRC_II_PUBLIC_FUNCTIONS: frozenset[str] = frozenset(
    [
        # src_ii/sigma_schedule.py
        "resolution_shift",
        "time_snr_shift",
        "build_sigmas",
        "simple_scheduler",
        "build_sigma_schedule",
        "const_noise_scaling",
        "const_inverse_noise_scaling",
        # src_ii/solver.py
        "euler_solve",
        "to_d",
        # src_ii/forward.py
        "nfe",
        "denoise",
        # src_ii/guided_denoiser.py
        "make_guided_denoiser",
        "pad_and_batch_cond",
        # src_ii/stats.py
        "spearman_rank_correlation",
        "sigma_for_step",
        "finite_differences",       # proposed -- not yet in stats.py
        "running_average",          # proposed
        "sliding_std",              # proposed
        # src_ii/reward_functions.py
        "pinkify_score",
        "thisnotthat_score_gpu",
        "pairwise_preference",
        # src_ii/vae_utils.py
        "decode_latent_to_pil",
        "load_vae",
        # src_ii/rendering.py (proposed -- does not exist yet)
        "save_tensor_as_png",
        "tensor_to_pil",
        "make_false_color_diff",
        "save_false_color_diff",
        "compute_per_channel_pixel_stats",
        "compute_spatial_autocorrelation",
    ]
)

# Name collision exceptions: (filename_stem, function_name)
# These cover functions in scripts_ii/ that share a name with the src_ii
# public API but are grandfathered until the canonical module exists and
# the script is updated to import from it.

NAME_COLLISION_EXCEPTIONS: set[tuple[str, str]] = {
    # All previously grandfathered name collisions have been resolved:
    # validate_packed_vs_serial.py no longer defines save_tensor_as_png,
    # compute_spatial_autocorrelation, or compute_per_channel_pixel_stats
    # inline -- it imports them from src_ii.rendering.
}


# ---------------------------------------------------------------------------
# Check 3: src/futudiffu/ freeze enforcement
# ---------------------------------------------------------------------------
#
# src/futudiffu/ is FROZEN. No new imports from this package in src_ii/ or
# scripts_ii/. This check uses AST parsing to detect real import statements
# (not comments or docstrings) and regex for sys.path manipulations.
#
# Exception format: (directory, filename, module_or_pattern)
#   directory: "src_ii" or "scripts_ii"
#   filename: just the filename (e.g. "btrm_model.py")
#   module_or_pattern: the futudiffu submodule imported (e.g. "futudiffu.btrm")
#                      or "sys.path" for path manipulation violations
#
# Each entry must include a comment. Remove only after migration to src_ii.

SRC_FREEZE_EXCEPTIONS: list[tuple[str, str, str]] = [
    # -------------------------------------------------------------------
    # src_ii/ -- these modules wrap frozen futudiffu primitives and will
    # be migrated when the underlying functionality is ported to src_ii.
    # -------------------------------------------------------------------

    # model_loading.py: loads NextDiT architecture, FP8 conversion, SageAttention config.
    # Remove when: src_ii has its own model loading that doesn't need futudiffu internals.
    ("src_ii", "model_loading.py", "futudiffu.diffusion_model"),
    ("src_ii", "model_loading.py", "futudiffu.fp8"),
    ("src_ii", "model_loading.py", "futudiffu.sage_attention"),

    # btrm_model.py: compound model wrapping ScoreUnembedder, LoRA, HiddenCapture.
    # Remove when: ScoreUnembedder, LoRA, HiddenCapture are ported to src_ii.
    ("src_ii", "btrm_model.py", "futudiffu.btrm"),
    ("src_ii", "btrm_model.py", "futudiffu.lora"),
    ("src_ii", "btrm_model.py", "futudiffu.training_utils"),
    ("src_ii", "btrm_model.py", "futudiffu.diffusion_model"),

    # btrm_training.py: loss functions from futudiffu.btrm.
    # Remove when: bradley_terry_loss, logsquare_regularizer ported to src_ii.
    ("src_ii", "btrm_training.py", "futudiffu.btrm"),

    # dataset_generator.py: uses DatasetWriter from futudiffu.dataset_v2.
    # Remove when: dataset_v2 ported to src_ii.
    ("src_ii", "dataset_generator.py", "futudiffu.dataset_v2"),

    # attention_capture.py: monkey-patches futudiffu.attention for capture.
    # Remove when: attention module ported to src_ii.
    ("src_ii", "attention_capture.py", "futudiffu.attention"),
    ("src_ii", "attention_capture.py", "futudiffu.diffusion_model"),

    # vae_utils.py: wraps futudiffu.vae for encode/decode.
    # Remove when: VAE loading ported to src_ii.
    ("src_ii", "vae_utils.py", "futudiffu.vae"),

    # -------------------------------------------------------------------
    # scripts_ii/ -- operational scripts that import from frozen package.
    # Each needs migration to use src_ii wrappers instead.
    # -------------------------------------------------------------------

    # sweep_rtheta_lr.py: uses text_encoder, sampling, lora, attention, diffusion_model.
    # Remove when: sweep script migrated to src_ii model_loading + forward.
    ("scripts_ii", "sweep_rtheta_lr.py", "futudiffu.text_encoder"),
    ("scripts_ii", "sweep_rtheta_lr.py", "futudiffu.sampling"),
    ("scripts_ii", "sweep_rtheta_lr.py", "futudiffu.lora"),
    ("scripts_ii", "sweep_rtheta_lr.py", "futudiffu.attention"),
    ("scripts_ii", "sweep_rtheta_lr.py", "futudiffu.diffusion_model"),
    ("scripts_ii", "sweep_rtheta_lr.py", "sys.path"),

    # verify_btrm_persistence.py: imports ScoreUnembedder from futudiffu.btrm.
    # Remove when: uses src_ii.btrm_model instead.
    ("scripts_ii", "verify_btrm_persistence.py", "futudiffu.btrm"),
    ("scripts_ii", "verify_btrm_persistence.py", "sys.path"),

    # generate_btrm_dataset.py: uses PROMPT_TEMPLATES and InferenceClient.
    # Remove when: prompt templates moved to src_ii, client wrapped.
    ("scripts_ii", "generate_btrm_dataset.py", "futudiffu.btrm_dataset"),
    ("scripts_ii", "generate_btrm_dataset.py", "futudiffu.client"),
    ("scripts_ii", "generate_btrm_dataset.py", "sys.path"),

    # run03_btrm_training.py: uses InferenceClient and DatasetReader/Writer.
    # Remove when: migrated to src_ii wrappers.
    ("scripts_ii", "run03_btrm_training.py", "futudiffu.client"),
    ("scripts_ii", "run03_btrm_training.py", "futudiffu.dataset_v2"),
    ("scripts_ii", "run03_btrm_training.py", "sys.path"),

    # merge_v2_datasets.py: uses INDEX_SCHEMA, _PARQUET_WRITE_KWARGS from dataset_v2.
    # Remove when: dataset_v2 ported to src_ii.
    ("scripts_ii", "merge_v2_datasets.py", "futudiffu.dataset_v2"),
    ("scripts_ii", "merge_v2_datasets.py", "sys.path"),

    # attention_interpretability.py: uses text_encoder, sampling, attention, diffusion_model.
    # Remove when: migrated to src_ii model_loading.
    ("scripts_ii", "attention_interpretability.py", "futudiffu.text_encoder"),
    ("scripts_ii", "attention_interpretability.py", "futudiffu.sampling"),
    ("scripts_ii", "attention_interpretability.py", "futudiffu.attention"),
    ("scripts_ii", "attention_interpretability.py", "futudiffu.diffusion_model"),
    ("scripts_ii", "attention_interpretability.py", "sys.path"),

    # sweep_rtheta_hparams.py: uses text_encoder, sampling, lora, attention, diffusion_model.
    # Remove when: migrated to src_ii model_loading.
    ("scripts_ii", "sweep_rtheta_hparams.py", "futudiffu.text_encoder"),
    ("scripts_ii", "sweep_rtheta_hparams.py", "futudiffu.sampling"),
    ("scripts_ii", "sweep_rtheta_hparams.py", "futudiffu.lora"),
    ("scripts_ii", "sweep_rtheta_hparams.py", "futudiffu.attention"),
    ("scripts_ii", "sweep_rtheta_hparams.py", "futudiffu.diffusion_model"),
    ("scripts_ii", "sweep_rtheta_hparams.py", "sys.path"),

    # validate_packed_vs_serial.py: uses protocol, client.
    # Remove when: migrated to src_ii wrappers.
    ("scripts_ii", "validate_packed_vs_serial.py", "futudiffu.protocol"),
    ("scripts_ii", "validate_packed_vs_serial.py", "futudiffu.client"),
    ("scripts_ii", "validate_packed_vs_serial.py", "sys.path"),

    # validate_v2_dataset.py: uses dataset_v2, client.
    # Remove when: dataset_v2 ported to src_ii.
    ("scripts_ii", "validate_v2_dataset.py", "futudiffu.dataset_v2"),
    ("scripts_ii", "validate_v2_dataset.py", "futudiffu.client"),
    ("scripts_ii", "validate_v2_dataset.py", "sys.path"),

    # train_pinkify_btrm.py: uses text_encoder, sampling.
    # Remove when: migrated to src_ii model_loading.
    ("scripts_ii", "train_pinkify_btrm.py", "futudiffu.text_encoder"),
    ("scripts_ii", "train_pinkify_btrm.py", "futudiffu.sampling"),
    ("scripts_ii", "train_pinkify_btrm.py", "sys.path"),

    # attention_adapter_diff.py: uses lora, text_encoder, sampling, attention, diffusion_model.
    # Remove when: migrated to src_ii model_loading.
    ("scripts_ii", "attention_adapter_diff.py", "futudiffu.lora"),
    ("scripts_ii", "attention_adapter_diff.py", "futudiffu.text_encoder"),
    ("scripts_ii", "attention_adapter_diff.py", "futudiffu.sampling"),
    ("scripts_ii", "attention_adapter_diff.py", "futudiffu.attention"),
    ("scripts_ii", "attention_adapter_diff.py", "futudiffu.diffusion_model"),
    ("scripts_ii", "attention_adapter_diff.py", "sys.path"),

    # validate_trajectory.py: uses text_encoder, attention.
    # Remove when: migrated to src_ii model_loading.
    ("scripts_ii", "validate_trajectory.py", "futudiffu.text_encoder"),
    ("scripts_ii", "validate_trajectory.py", "futudiffu.attention"),
    ("scripts_ii", "validate_trajectory.py", "sys.path"),

    # render_attention_maps.py: sys.path manipulation only (no direct futudiffu import).
    # Remove when: script no longer needs src/ on path.
    ("scripts_ii", "render_attention_maps.py", "sys.path"),

    # render_attention_maps_v2.py: sys.path manipulation only.
    # Remove when: script no longer needs src/ on path.
    ("scripts_ii", "render_attention_maps_v2.py", "sys.path"),

    # backfill_v2_hashes.py: sys.path manipulation only.
    # Remove when: script no longer needs src/ on path.
    ("scripts_ii", "backfill_v2_hashes.py", "sys.path"),

    # generate_preference_labels.py: sys.path manipulation only.
    # Remove when: script no longer needs src/ on path.
    ("scripts_ii", "generate_preference_labels.py", "sys.path"),

    # measure_prompt_tokens.py: sys.path manipulation only.
    # Remove when: script no longer needs src/ on path.
    ("scripts_ii", "measure_prompt_tokens.py", "sys.path"),

    # score_distribution_comparison.py: sys.path manipulation only.
    # Remove when: script no longer needs src/ on path.
    ("scripts_ii", "score_distribution_comparison.py", "sys.path"),

    # migrate_v1_into_v2.py: sys.path manipulation only (Windows hardcoded path).
    # Remove when: script no longer needs src/ on path.
    ("scripts_ii", "migrate_v1_into_v2.py", "sys.path"),

    # render_comparison.py: sys.path manipulation only (adds both src/ and repo root).
    # Remove when: script no longer needs src/ on path.
    ("scripts_ii", "render_comparison.py", "sys.path"),
]

_SRC_FREEZE_EXCEPTION_SET: set[tuple[str, str, str]] = set(SRC_FREEZE_EXCEPTIONS)


def _find_frozen_import_violations(
    path: Path,
    directory_label: str,
) -> list[tuple[str, int, str]]:
    """Check a file for imports from the frozen src/futudiffu/ package.

    Uses AST parsing to detect real import statements (not comments or
    docstrings), plus regex for sys.path manipulations.

    Args:
        path: Path to the .py file.
        directory_label: "src_ii" or "scripts_ii" (for exception lookup).

    Returns:
        List of (violation_id, line_number, message) for ungrandfathered violations.
    """
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [("<read_error>", 0, f"Could not read {path}: {exc}")]

    filename = path.name
    violations: list[tuple[str, int, str]] = []

    # --- AST-based import detection ---
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        # If we can't parse, fall back to regex-only (better than nothing).
        tree = None

    if tree is not None:
        for node in ast.walk(tree):
            module_name = None

            if isinstance(node, ast.ImportFrom) and node.module:
                # `from futudiffu.X import Y`
                if node.module == "futudiffu" or node.module.startswith("futudiffu."):
                    module_name = node.module
            elif isinstance(node, ast.Import):
                # `import futudiffu.X` or `import futudiffu`
                for alias in node.names:
                    if alias.name == "futudiffu" or alias.name.startswith("futudiffu."):
                        module_name = alias.name
                        break

            if module_name is not None:
                # Normalize to top two levels: "futudiffu.X.Y.Z" -> "futudiffu.X"
                parts = module_name.split(".")
                normalized = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]

                if (directory_label, filename, normalized) in _SRC_FREEZE_EXCEPTION_SET:
                    continue

                violations.append((
                    f"frozen_import:{normalized}",
                    node.lineno,
                    f"Import from frozen package: '{module_name}' "
                    f"(src/futudiffu/ is frozen -- use src_ii equivalents)",
                ))

    # --- Regex-based sys.path detection ---
    # Matches: sys.path.insert(0, .../src) or sys.path.append(.../src)
    # Also matches sys.path.insert(0, os.path.join(..., "src"))
    syspath_pattern = re.compile(
        r'sys\.path\.(?:insert|append)\s*\([^)]*["\']src["\']'
        r'|sys\.path\.(?:insert|append)\s*\([^)]*[/\\]src["\'\)]'
        r'|sys\.path\.(?:insert|append)\s*\([^)]*"src"'
        r"|sys\.path\.(?:insert|append)\s*\([^)]*\bsrc\b",
        re.MULTILINE,
    )
    for match in syspath_pattern.finditer(source):
        line_no = source[: match.start()].count("\n") + 1
        if (directory_label, filename, "sys.path") in _SRC_FREEZE_EXCEPTION_SET:
            continue
        violations.append((
            "frozen_syspath",
            line_no,
            f"sys.path manipulation adding src/ to path "
            f"(enables imports from frozen src/futudiffu/)",
        ))

    return violations


# ---------------------------------------------------------------------------
# Core checking logic (Checks 1 and 2)
# ---------------------------------------------------------------------------

def _find_violations_in_file(
    path: Path,
) -> list[tuple[str, int, str]]:
    """Check a single scripts_ii/ file for inlining pattern violations.

    Returns:
        List of (pattern_id, line_number, message) tuples for ungrandfathered
        violations.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [(f"<read_error>", 0, f"Could not read {path}: {exc}")]

    filename = path.name
    violations: list[tuple[str, int, str]] = []

    for pattern_id, regex, message in FORBIDDEN_PATTERNS:
        if (filename, pattern_id) in _EXCEPTION_SET:
            continue
        for match in re.finditer(regex, text):
            line_no = text[: match.start()].count("\n") + 1
            violations.append((pattern_id, line_no, message))

    return violations


def _extract_top_level_function_names(path: Path) -> list[tuple[str, int]]:
    """Parse a Python file and return all top-level function names + line numbers.

    Uses ast.parse so it handles multi-line definitions correctly.
    Private functions (starting with _) are excluded from collision checking.

    Returns:
        List of (function_name, line_number) for public top-level defs.
    """
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, SyntaxError):
        return []

    results = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            results.append((node.name, node.lineno))
    return results


def _find_name_collisions_in_file(
    path: Path,
) -> list[tuple[str, int, str]]:
    """Check a single scripts_ii/ file for name collisions with src_ii public API.

    Returns:
        List of (function_name, line_number, message) for ungrandfathered collisions.
    """
    filename = path.name
    collisions: list[tuple[str, int, str]] = []

    for func_name, line_no in _extract_top_level_function_names(path):
        if func_name not in SRC_II_PUBLIC_FUNCTIONS:
            continue
        if (filename, func_name) in NAME_COLLISION_EXCEPTIONS:
            continue
        collisions.append(
            (
                func_name,
                line_no,
                f"Name collision: '{func_name}' is a public function in src_ii "
                f"(or proposed for src_ii/rendering.py). "
                f"Scripts must import from the canonical module, not redefine.",
            )
        )

    return collisions


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Run all pokayoke checks. Returns exit code (0 = pass, 1 = fail)."""
    if not SCRIPTS_II_DIR.exists():
        print(f"POKAYOKE ERROR: scripts_ii/ directory not found at {SCRIPTS_II_DIR}")
        return 1

    scripts_ii_files = sorted(SCRIPTS_II_DIR.glob("*.py"))
    src_ii_files = sorted(SRC_II_DIR.glob("*.py")) if SRC_II_DIR.exists() else []

    if not scripts_ii_files:
        print("POKAYOKE WARNING: No .py files found in scripts_ii/")

    # --- Check 1: Inlining violations (scripts_ii/ only) ---
    all_inline_violations: list[tuple[Path, str, int, str]] = []
    all_collision_violations: list[tuple[Path, str, int, str]] = []

    for py_file in scripts_ii_files:
        inline_viols = _find_violations_in_file(py_file)
        for pattern_id, line_no, message in inline_viols:
            all_inline_violations.append((py_file, pattern_id, line_no, message))

        # --- Check 2: Name collision violations (scripts_ii/ only) ---
        collision_viols = _find_name_collisions_in_file(py_file)
        for func_name, line_no, message in collision_viols:
            all_collision_violations.append((py_file, func_name, line_no, message))

    # --- Check 3: Frozen import violations (src_ii/ AND scripts_ii/) ---
    all_freeze_violations: list[tuple[Path, str, int, str]] = []

    for py_file in src_ii_files:
        freeze_viols = _find_frozen_import_violations(py_file, "src_ii")
        for viol_id, line_no, message in freeze_viols:
            all_freeze_violations.append((py_file, viol_id, line_no, message))

    for py_file in scripts_ii_files:
        freeze_viols = _find_frozen_import_violations(py_file, "scripts_ii")
        for viol_id, line_no, message in freeze_viols:
            all_freeze_violations.append((py_file, viol_id, line_no, message))

    total_violations = (
        len(all_inline_violations)
        + len(all_collision_violations)
        + len(all_freeze_violations)
    )

    # --- Report ---
    if total_violations == 0:
        n_inline_ex = len(POKAYOKE_EXCEPTIONS)
        n_collision_ex = len(NAME_COLLISION_EXCEPTIONS)
        n_freeze_ex = len(SRC_FREEZE_EXCEPTIONS)
        print(
            f"POKAYOKE PASS: No ungrandfathered violations "
            f"({n_inline_ex} inline, {n_collision_ex} name-collision, "
            f"{n_freeze_ex} frozen-import exceptions grandfathered "
            f"-- shrink these as violations are fixed)"
        )
        return 0

    print(
        f"POKAYOKE FAIL: {total_violations} ungrandfathered violation(s)"
    )

    if all_inline_violations:
        print(f"\n-- Inlining violations ({len(all_inline_violations)}) --")
        for py_file, pattern_id, line_no, message in all_inline_violations:
            print(f"  {py_file.name}:{line_no}: [{pattern_id}] {message}")

    if all_collision_violations:
        print(f"\n-- Name collision violations ({len(all_collision_violations)}) --")
        for py_file, func_name, line_no, message in all_collision_violations:
            print(f"  {py_file.name}:{line_no}: [name_collision:{func_name}] {message}")

    if all_freeze_violations:
        print(f"\n-- Frozen import violations ({len(all_freeze_violations)}) --")
        for py_file, viol_id, line_no, message in all_freeze_violations:
            rel = py_file.relative_to(REPO_ROOT)
            print(f"  {rel}:{line_no}: [{viol_id}] {message}")

    print(
        "\nTo grandfather a violation until it is fixed, add an entry to:\n"
        "  POKAYOKE_EXCEPTIONS (inlining)\n"
        "  NAME_COLLISION_EXCEPTIONS (name collisions)\n"
        "  SRC_FREEZE_EXCEPTIONS (frozen imports)\n"
        "with a comment explaining what will remove it."
    )

    return 1


if __name__ == "__main__":
    sys.exit(main())

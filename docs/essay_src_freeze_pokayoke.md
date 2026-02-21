# src/futudiffu/ Freeze Pokayoke Guard

## What This Is

A mechanical enforcement guard added to `scripts/pokayoke_inline_check.py`
(Check 3) that prevents new imports from the frozen `src/futudiffu/` package
in any file under `src_ii/` or `scripts_ii/`.

## What It Checks

The guard detects three categories of violation:

1. **Direct imports** (`from futudiffu.X import Y` or `import futudiffu.X`):
   Detected via AST parsing of the Python source. This means comments and
   docstrings that mention `futudiffu` are not flagged -- only real import
   statements in executable code.

2. **Module-level imports** (`import futudiffu.X as alias`): Also detected
   via AST, covering both `ast.Import` and `ast.ImportFrom` node types.

3. **sys.path manipulation** (`sys.path.insert(0, .../src)`): Detected via
   regex, since this is not an import statement but the enabling mechanism
   that makes futudiffu importable. Files that add `src/` to `sys.path`
   are flagged even if they don't directly import futudiffu, because the
   path manipulation exists solely to enable such imports.

## Scope

The guard scans:
- All `.py` files in `src_ii/`
- All `.py` files in `scripts_ii/`

It does NOT scan `src/futudiffu/` itself (that's the frozen package),
`scripts/` (legacy scripts that predate the split), or `tests/`.

## Grandfathering

All 65 existing violations are grandfathered in `SRC_FREEZE_EXCEPTIONS`,
a list of `(directory, filename, module_or_pattern)` tuples. Each entry
has a comment explaining what it tracks and when it can be removed.

The exception list should shrink monotonically as imports are migrated
to `src_ii` equivalents. Adding a new entry requires a comment.

### Breakdown of grandfathered violations

**src_ii/ (13 exceptions across 6 files)**:
These are `src_ii` modules that wrap frozen `futudiffu` primitives. They
form the boundary layer and will be the last to migrate.

| File | Modules imported |
|------|-----------------|
| `model_loading.py` | diffusion_model, fp8, sage_attention |
| `btrm_model.py` | btrm, lora, training_utils, diffusion_model |
| `btrm_training.py` | btrm |
| `dataset_generator.py` | dataset_v2 |
| `attention_capture.py` | attention, diffusion_model |
| `vae_utils.py` | vae |

**scripts_ii/ (52 exceptions across 19 files)**:
Mix of direct futudiffu imports and sys.path manipulations.

- 11 files have direct futudiffu imports (text_encoder, sampling, lora, attention, diffusion_model, client, dataset_v2, protocol, btrm, btrm_dataset)
- 17 files have sys.path manipulations adding `src/` to the path
- 6 files have sys.path only (no direct imports, but the path manipulation enables potential future imports)

## Running the Guard

```bash
# Full pokayoke check (all 3 check groups):
python scripts/pokayoke_inline_check.py

# Expected output when passing:
# POKAYOKE PASS: No ungrandfathered violations (1 inline, 0 name-collision, 65 frozen-import exceptions grandfathered -- shrink these as violations are fixed)

# Expected output when a new violation is introduced:
# POKAYOKE FAIL: 1 ungrandfathered violation(s)
# -- Frozen import violations (1) --
#   src_ii/new_module.py:5: [frozen_import:futudiffu.lora] Import from frozen package: 'futudiffu.lora' (src/futudiffu/ is frozen -- use src_ii equivalents)
```

## Verification

The guard was tested with synthetic canary files in both `src_ii/` and
`scripts_ii/` that contained:
- `from futudiffu.sampling import make_rope_cache` -- caught as `frozen_import:futudiffu.sampling`
- `import futudiffu.attention as attn_mod` -- caught as `frozen_import:futudiffu.attention`
- `sys.path.insert(0, "src")` -- caught as `frozen_syspath`

All three violation types were detected with correct file paths, line
numbers, and human-readable messages. Canary files were cleaned up after
testing.

## Design Decisions

**AST parsing for imports, regex for sys.path**: Import detection uses
Python's `ast` module to walk the parse tree. This avoids false positives
from comments and docstrings that mention futudiffu (e.g., the module-level
docstrings in `sigma_schedule.py` and `solver.py` that say "IMPORTS nothing
from futudiffu"). The `sys.path` check uses regex because `sys.path.insert`
is a method call, not an import statement, and AST-based detection of
arbitrary method calls would be fragile.

**Normalized module matching**: Import module names are normalized to the
top two dotted components (`futudiffu.X`). This means `from futudiffu.btrm
import ScoreUnembedder` and `from futudiffu.btrm import bradley_terry_loss`
are both matched by the single exception `("src_ii", "btrm_training.py",
"futudiffu.btrm")`. This keeps the exception list manageable without
losing precision.

**Integrated into existing script**: Rather than creating a separate
script, the guard was added as Check 3 within the existing
`pokayoke_inline_check.py`. This ensures a single command runs all boundary
checks and a single exit code reports overall compliance.

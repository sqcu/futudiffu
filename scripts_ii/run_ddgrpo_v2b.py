r"""DDGRPO v2b: Higher K, higher LR, shorter run with sign agreement tracking.

Thin wrapper over run_ddgrpo_v2. Overrides:
  B=2, K=3 (6 rollouts/iter), LR=6e-5, N_ITERS=50, OUTPUT_DIR=ddgrpo_v2b.

The hypothesis: K=3 gives 6 rollouts per prompt group, improving advantage
estimates (3 comparisons vs 2). Higher LR compensates for shorter run.

Execution:
  PYTHONUNBUFFERED=1 .venv/Scripts/python.exe ^
      F:\dox\repos\ai\futudiffu\scripts_ii\run_ddgrpo_v2b.py
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

# Import everything from v2 — all functions are module-level
import scripts_ii.run_ddgrpo_v2 as v2

# ---------------------------------------------------------------------------
# Override hyperparams
# ---------------------------------------------------------------------------

v2.B = 2
v2.K = 3           # 6 rollouts/iter (was 4)
v2.LR = 6e-5       # 3x v2's LR
v2.N_ITERS = 50    # shorter run
v2.N_STEPS = 20    # unchanged
v2.OUTPUT_DIR = REPO_ROOT / "training_output" / "ddgrpo_v2b"

if __name__ == "__main__":
    try:
        v2.main()
    except Exception:
        import traceback
        tb = traceback.format_exc()
        v2._log(f"FATAL EXCEPTION:\n{tb}")
        raise

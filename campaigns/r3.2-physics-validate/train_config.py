"""
R3.2: train the physics-based mapping on the R3.1 ground truth (task 0c-1, 2026-10-09).

BO search box, initial point and run settings. The conditions, ground-truth
coefficients, density, N and seeds are in ``../r3.1-gt/gt_config.py``.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r3.1-gt"))

from gt_config import (  # noqa: E402,F401
    ALL_CONDITIONS, DENSITY, GROUND_TRUTH, HELDOUT, N, TRAIN,
)

# Search box. Lower bounds of A, a_m, delta keep U0 >= 0.5, m >= 4, n >= 5.5,
# n - m >= 1 for every condition above (smallest g = 1.375 at k = 0.9,
# L_bridge = 120); tests/test_physics_map.py checks this on a grid.
# Initial point: datasets.py defaults for k, A, K_s; a_m = delta such that
# n = 12, m = 6 at L_bridge = 20 and k = 0.76.
PARAM_CFG = {
    "k":     {"bounds": (0.40, 0.90), "init": 0.76},
    "A":     {"bounds": (0.85, 2.50), "init": 2.0},
    "K_s":   {"bounds": (0.00, 0.10), "init": 0.05},
    "a_m":   {"bounds": (2.91, 4.00), "init": 3.2727},
    "delta": {"bounds": (1.09, 4.00), "init": 3.2727},
}

BO_SEED = 0       # torch seed of run_bo; simulations use the run_simulation default seed (42)
N_ITERS = 99      # acquisitions: 1 initial + 99 = 100 evaluations in total, resumes included
METRIC = "shift_rmse"   # default shift_rmse parameters; compare over the curve overlap

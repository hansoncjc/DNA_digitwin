"""
R3.2: physics-based mapping trained on a virtual ground truth (task 0c-1, 2026-10-09).

Single source for the condition set, the ground-truth coefficients, the BO
search box and the run settings. Used by ``make_ground_truth.py``,
``qc_ground_truth.py``, ``run_r32.py`` and ``tests/test_physics_map.py``.

Mapping (``datasets.physics_map``)::

    r0/sigma = 1 + k (l + 0.34 L_bridge) / d,  l = 32.04 nm, d = 24.6 nm
    g = r0 / (r0 - 1);  m = a_m g;  n = (a_m + delta) g
    U0 = A (C_chol / 100) (1 + K_s C_NaCl)

Density is fixed at 0.005 sigma^-3, the value used in the simulations of
Chiang et al. 2025.
"""
from itertools import product

# ---------------------------------------------------------------- conditions
# Training: full factorial, 2 bridge lengths x 3 loadings x 4 salt levels = 24.
# Levels from the Chiang et al. 2025 / Huat data design (C_chol 20-140,
# NaCl 0-40 mM, bridges 20/40/80 bp); C_chol = 20 left out (low-loading
# regime the linear U0 form does not describe).
TRAIN_L_BRIDGE = (20.0, 80.0)          # bp
TRAIN_C_CHOL = (60.0, 100.0, 140.0)    # molecules / particle
TRAIN_C_NACL = (0.0, 10.0, 20.0, 40.0)  # mM

TRAIN = [
    {"id": f"d{i}", "L_bridge": L, "C_chol": c, "C_NaCl": s, "role": "train", "note": ""}
    for i, (L, c, s) in enumerate(product(TRAIN_L_BRIDGE, TRAIN_C_CHOL, TRAIN_C_NACL))
]

# Held-out (R5.1 prediction); not used in training.
HELDOUT = [
    {"id": "h0", "L_bridge": 40.0, "C_chol": 100.0, "C_NaCl": 20.0, "role": "heldout", "note": "bridge interpolation"},
    {"id": "h1", "L_bridge": 20.0, "C_chol": 80.0, "C_NaCl": 30.0, "role": "heldout", "note": "loading + salt interpolation"},
    {"id": "h2", "L_bridge": 80.0, "C_chol": 120.0, "C_NaCl": 10.0, "role": "heldout", "note": "loading + salt interpolation"},
    {"id": "h3", "L_bridge": 120.0, "C_chol": 100.0, "C_NaCl": 20.0, "role": "heldout", "note": "bridge extrapolation"},
    {"id": "h4", "L_bridge": 20.0, "C_chol": 100.0, "C_NaCl": 50.0, "role": "heldout", "note": "salt extrapolation"},
]

ALL_CONDITIONS = TRAIN + HELDOUT

# ---------------------------------------------------------------- coefficients
# Ground truth: k puts every r0 inside the inverse-design FCC box [2, 3.5];
# a_m, delta give n, m = 12.6, 6.3 at L_bridge = 20 (close to the old 12-6).
GROUND_TRUTH = {"k": 0.65, "A": 1.2, "K_s": 0.07, "a_m": 3.2, "delta": 3.2}

# Search box. Lower bounds of A, a_m, delta keep U0 >= 0.5, m >= 4, n >= 5.5,
# n - m >= 1 for every condition above (smallest g = 1.375 at k = 0.9,
# L_bridge = 120); tests/test_physics_map.py checks this on a grid.
# Initial point: datasets.py defaults for k, A, K_s; a_m = delta such that
# n = 12, m = 6 at L_bridge = 20 and k = 0.76.
PARAM_CFG = {
    "global": {
        "k":     {"bounds": (0.40, 0.90), "init": 0.76},
        "A":     {"bounds": (0.85, 2.50), "init": 2.0},
        "K_s":   {"bounds": (0.00, 0.10), "init": 0.05},
        "a_m":   {"bounds": (2.91, 4.00), "init": 3.2727},
        "delta": {"bounds": (1.09, 4.00), "init": 3.2727},
    },
    "local": {},
}

# ---------------------------------------------------------------- run settings
DENSITY = 0.005   # sigma^-3, Chiang et al. 2025 simulations
N = 5000          # forward model: everything else at function defaults
GT_SEED = 1       # ground truth; BO evaluations use the run_simulation default (42)
BO_SEED = 0       # torch seed of run_bo
N_ITERS = 200     # upper bound; the master walltime ends the run earlier

METRIC = "shift_rmse"   # default shift_rmse parameters; compare over the curve overlap

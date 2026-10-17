"""
R5.1: forward prediction of S(q) with the trained physics-based mapping (2026-10-14).

Prediction only: the mapping trained in R3.2 (eval 56) is applied to
conditions chosen by the author; no loss against a ground truth is computed.
Training conditions (d0-d23) are not run: R3.2 already evaluated eval 56
there. Held-out cells on the grids (h0, h3) were first left out too
(author, 2026-10-14) and added on 2026-10-16 (author) as p14, p15, so the
figures can show the prediction next to their ground truth.

Coefficients: R3.2 eval 56 (best total loss 0.607), full precision from the
R3.2 ``Optimization_Results/bo_trajectory.csv`` (Desktop copy
``11_R32_analysis/``; rounded in ``../r3.2-physics-validate/analysis/r32_evals.csv``).

Conditions, two grids (author, 2026-10-14):
  part 1: L_bridge = 80, axes C_anchor x C_NaCl. The R3.1 training grid
          (C_anchor 60, 100, 140 x C_NaCl 0, 10, 20, 40) extended by a row
          C_anchor = 45 and a column C_NaCl = 60; the 8 new cells.
  part 2: C_anchor = 100, axes C_NaCl (0, 10, 20, 40) x L_bridge (20, 40, 80,
          120); the 6 cells without a ground truth (L_bridge 20 and 80 are
          d4-d7, d16-d19), then the 2 held-out cells (40, 100, 20) = h0 and
          (120, 100, 20) = h3.

Ids follow run order: p0-p13 (submitted 2026-10-14) are the grid cells with
no ground truth, p14-p15 the grid cells with a held-out ground truth, so the
ids and Slurm array indices of the runs already made do not change.

Forward model: function defaults, N = 5000, density 0.005, run_simulation's
default seed (42, as in the R3.2 evaluations), one run per condition, S(q) on
the CPU (convert_to_SAXS_fft defaults, C26).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r3.1-gt"))

from gt_config import ALL_CONDITIONS as GT_CONDITIONS  # noqa: E402
from gt_config import DENSITY, N  # noqa: E402,F401
from gt_config import TRAIN_C_ANCHOR, TRAIN_C_NACL, TRAIN_L_BRIDGE  # noqa: E402

# ---------------------------------------------------------------- coefficients
# R3.2 eval 56, bo_trajectory.csv (every row of eval 56 carries the same values).
COEFFS = {
    "k": 0.6520927389752293,
    "A": 1.2028501815438974,
    "K_s": 0.07251128398087511,
    "a_m": 3.2087563613999377,
    "delta": 3.3893817640184665,
}
COEFFS_SOURCE = "R3.2 eval 56 (DNA_digitwin 2897707), Optimization_Results/bo_trajectory.csv"

# ---------------------------------------------------------------- conditions
PART1_L_BRIDGE = 80.0
PART1_C_ANCHOR = (45.0, 60.0, 100.0, 140.0)
PART1_C_NACL = (0.0, 10.0, 20.0, 40.0, 60.0)

PART2_C_ANCHOR = 100.0
PART2_L_BRIDGE = (20.0, 40.0, 80.0, 120.0)
PART2_C_NACL = (0.0, 10.0, 20.0, 40.0)

_GT_ROLES = {(c["L_bridge"], c["C_anchor"], c["C_NaCl"]): c["role"] for c in GT_CONDITIONS}


def training_range_tag(L_bridge, C_anchor, C_NaCl):
    """'interp' inside the R3.1 training ranges, else 'extrap(<axes outside>)'."""
    out = [name for name, v, levels in (
        ("L_bridge", L_bridge, TRAIN_L_BRIDGE),
        ("C_anchor", C_anchor, TRAIN_C_ANCHOR),
        ("C_NaCl", C_NaCl, TRAIN_C_NACL),
    ) if not min(levels) <= v <= max(levels)]
    return f"extrap({','.join(out)})" if out else "interp"


def _grid():
    part1 = [(PART1_L_BRIDGE, c, s) for c in PART1_C_ANCHOR for s in PART1_C_NACL]
    part2 = [(L, PART2_C_ANCHOR, s) for L in PART2_L_BRIDGE for s in PART2_C_NACL]
    conds = []
    # first the cells with no ground truth (p0-p13), then those with a held-out one
    for gt_role in (None, "heldout"):
        for part, grid in (("part1", part1), ("part2", part2)):
            for L, c, s in grid:
                if _GT_ROLES.get((L, c, s)) != gt_role:
                    continue
                conds.append({
                    "id": f"p{len(conds)}", "L_bridge": L, "C_anchor": c, "C_NaCl": s,
                    "part": part, "range": training_range_tag(L, c, s),
                })
    return conds


#: Prediction conditions p0-p15 (Slurm array index = list index).
CONDITIONS = _grid()

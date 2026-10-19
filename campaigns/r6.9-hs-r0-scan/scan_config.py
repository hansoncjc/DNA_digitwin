"""
R6.9: fixed-point r0 scan next to the R6.6 best point (2026-10-18).

Question: R6.6 eval 67 (U0 1.8249, r0 1.3445, n 15, m 4) is a disordered
liquid with target-like droplets but a spacing about 14 % too large
(q1 ratio 0.856). Does lowering r0 to the target spacing keep it a liquid of
a few large droplets, or does it order / split into many small clusters?
(HS note §12: "Inference, not tested".)

Target r0: 1.3445 * 0.856 = 1.15 (from q1) or 1.3445 * 1.028 / 1.292 = 1.07
(from the g(r) first peak). No evaluation of R6.6, R6.8b or R6.8c sampled
U0 1.8-2.4, r0 1.07-1.2 at n 15, m 4.

Grid (author, 2026-10-18): r0 in {1.07, 1.11, 1.15, 1.20} x U0 in
{1.8, 2.1, 2.4}, n = 15, m = 4 (m_rel = 0), one seed; plus s0, the R6.6
eval-67 point itself, to check this driver against the BO run.

Forward model: function defaults (as R6.6), N = 5000, density 0.005,
run_simulation's default seed (42), S(q) on the CPU (convert_to_SAXS_fft
defaults, C26). Scored with shift_rmse against the R6.6 target and
METRIC_KWARGS ({} = lambda 1, as R6.6); M4 and the q1 ratio are written too,
so the loss at another lambda follows without rerunning.
"""
from pathlib import Path

TARGETS = Path(__file__).resolve().parents[1] / "inverse" / "inverse-targets"

# Same file as campaigns/inverse/r6.6-hs-fluid/config.py
# (md5 37c39120c6d44871953769456f8f3b75, cpu/cpu_info.json).
TARGET = TARGETS / "fluid_hs_rho0.763944_D23.21_pad4.309" / "cpu" / "target_sq.npy"
TARGET_MD5 = "37c39120c6d44871953769456f8f3b75"
N = 5000
DENSITY = 0.005
METRIC_KWARGS = {}

# R6.6 eval 67, full precision from Optimization_Results/eval_067/d0/sim_params_d0.csv
EVAL67 = {"U0": 1.8248825184973128, "r0": 1.3445141132218734, "n": 15.0, "m": 4.0}

N_EXP, M_EXP = 15.0, 4.0
R0_GRID = (1.07, 1.11, 1.15, 1.20)
U0_GRID = (1.8, 2.1, 2.4)

POINTS = [{"id": "s0", "role": "control (R6.6 eval 67)", **EVAL67}] + [
    {"id": f"s{1 + i * len(U0_GRID) + j}", "role": "scan",
     "U0": u0, "r0": r0, "n": N_EXP, "m": M_EXP}
    for i, r0 in enumerate(R0_GRID) for j, u0 in enumerate(U0_GRID)
]

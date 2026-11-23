"""
R7.1: one-step inverse design (physics-based endpoint) against the R6.5
target, the FCC crystallite at rho* = 0.05 (TIMELINE task 4.1).

BO over the experimental conditions x = (L_bridge, C_anchor, C_NaCl). Each
candidate goes through the trained mapping (R3.2 eval 56, mapping file
../../r3.2-physics-validate/mapping/r3.2_eval056.json, coefficients fixed) to
(r0, U0, n, m), is simulated with the R6 forward model and scored with
shift_rmse (defaults, lambda = 1), as R6.5.

Search box (author, 2026-11-22): 3D, salt 0-40 mM (the R3.2 training range).
Rule: for that salt range, the largest box in x whose image lies inside the
R6.5 search ranges of r0 [2, 3.5] and U0 [0.5, 8]:
  - L_bridge from r0 = 1 + k (l + 0.34 L_bridge) / d_si at r0 = 2 and 3.5
    (16.72 to 183.15 bp with these coefficients);
  - U0 = A (C_anchor / 100)(1 + K_s C_NaCl) grows with both C_anchor and salt,
    so C_anchor_lo puts U0 = 0.5 at the lowest salt and C_anchor_hi puts
    U0 = 8 at the highest (41.57 to 170.52). Every U0 in [0.5, 8] is still
    reached in the box; at a single salt only part of it is.
  The bounds are computed below from the mapping file, not copied by hand;
  tests/test_onestep.py maps the box back and checks it. n and m follow
  L_bridge alone (m/n = a_m / (a_m + delta) = 0.486) and stay inside the R6
  ranges over the whole box. Salt and C_anchor enter U0 only through
  C_anchor (1 + K_s C_NaCl): one direction of the box is degenerate (as in
  R6.10), so BO returns one point of a curve of equivalent conditions.

Initial point (author, 2026-11-22): (80 bp, 100, 20 mM), an R3.2 training
condition; it maps to r0 2.570, U0 2.947 (n 10.80, m 5.25), close to the R6
initial point (r0 2.5, U0 3). First batch = initial point + 9 Sobol points.

Batch BO as R6.5: batch_size 10, max_evals 100, bo_seed 0; simulation seed
42 (run_simulation default). GPU jobs on gpu-a40, 4 h each (author,
2026-11-22; settings as R6.4). S(q) on CPU on both sides, as R6.5.

    python inverse.py campaigns/inverse/r7.1-onestep-fcc0.05/config.py --check
    sbatch -J inv_r7.1 -o <run dir>/master_%j.out \\
        --export=ALL,INV_REPO=$REPO,INV_CONFIG=$REPO/campaigns/inverse/r7.1-onestep-fcc0.05/config.py \\
        $REPO/campaigns/inverse/inverse-example/inverse_parallel.sbatch
Submission steps: README.md next to this file.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mapping import load_mapping  # noqa: E402

TARGETS = HERE.parent / "inverse-targets"
MAPPING = REPO / "campaigns" / "r3.2-physics-validate" / "mapping" / "r3.2_eval056.json"

# R6.5 search ranges of the parameters the box is built from (r6.5-fcc0.05/config.py).
R6_R0 = (2.0, 3.5)
R6_U0 = (0.5, 8.0)
SALT = (0.0, 40.0)                      # mM (author, 2026-11-22)
INIT = {"L_bridge": 80.0, "C_anchor": 100.0, "C_NaCl": 20.0}


def condition_box(mapping, r0_range=R6_R0, u0_range=R6_U0, salt=SALT):
    """Box in x whose image lies in r0_range x u0_range (see the module docstring)."""
    c, k = mapping.coefficients, mapping.constants

    def bridge(r0):
        return (k["d_si"] * (r0 - 1.0) / c["k"] - k["fixed_linker_nm"]) / k["dsdna_rise_nm"]

    def anchor(U0, s):
        return k["c_anchor_ref"] * U0 / (c["A"] * (1.0 + c["K_s"] * s))

    return {
        "L_bridge": (bridge(r0_range[0]), bridge(r0_range[1])),
        "C_anchor": (anchor(u0_range[0], salt[0]), anchor(u0_range[1], salt[1])),
        "C_NaCl": (float(salt[0]), float(salt[1])),
    }


BOX = condition_box(load_mapping(MAPPING))

CONFIG = {
    "description": "R7.1 one-step: BO over (L_bridge, C_anchor, C_NaCl) through the R3.2 "
                   "eval-56 mapping; R6.5 target FCC rho*=0.05 (CPU S(q)); salt 0-40 mM, "
                   "box = image inside the R6.5 r0 and U0 ranges; batch BO 10 x 10 on gpu-a40",

    # md5 780fd41a9f25b73dba663037a2415a25 (cpu/cpu_info.json), as R6.5
    "target": str(TARGETS / "fcc_rho0.05_D57.6_pad1.738" / "cpu" / "target_sq.npy"),
    "N": 5017,                         # meta.json, as R6.5
    "density": 0.005,

    "mapping": str(MAPPING),
    "search": {name: {"bounds": BOX[name], "init": INIT[name]} for name in
               ("L_bridge", "C_anchor", "C_NaCl")},

    "max_evals": 100,                  # first batch included, resumes counted
    "batch_size": 10,
    "bo_seed": 0,

    "out_root": "/gscratch/zeelab/hanson/codes/DNA_lipid_silica/testing/inverse/"
                "r7/r7.1-onestep-fcc0.05/Optimization_Results",
    "gp_log_dir": None,                # <out_root>/gp_log

    "potential": "mie_r0",
    "metric": "shift_rmse",
    "scattering": "saxsfft",

    "mc_dfm_root": "/gscratch/zeelab/hanson/codes/DNA_lipid_silica/MC-DFM",

    # One gpu-a40 job per evaluation, as R6.4 (r6.4-fcc0.10/config.py): about
    # 2.3 h at worst on an A40, 4 h limit.
    "parallel": True,
    "parallel_cfg": {
        "partition": "gpu-a40",
        "account": "zeelab",
        "mem": "10G",
        "time": "04:00:00",
        "module_loads": ["cuda/11.8"],
        "venv_activate": "/gscratch/zeelab/hanson/codes/HOOMD-start/hoomd-venv/bin/activate",
        "max_job_retries": 2,
        "poll_interval": 30.0,
        "max_wait": 6 * 3600,          # from submission: 4 h job + 2 h in the queue
    },
}

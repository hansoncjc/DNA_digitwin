"""
R7.1f: one-step inverse design (physics-based endpoint) against the R6.6
target, the hard-sphere fluid droplets at phi = 0.4, with lambda_shift = 2
(spacing term weighted 2 times the default). The one-step counterpart of R6.8c (r6.8c-hs-lam2/config.py).

The R7 HS campaigns are one set (author, 2026-11-23): R7.1d (lambda = 1,
counterpart of R6.6), R7.1e (lambda = 0, R6.8b) and R7.1f (lambda = 2,
R6.8c), discussed by lambda. The target spacing (r0 about 1.16, R6.9) is out
of the mapping's reach (r0 >= 1.894 in the box), as in R7.1d.

Rule: each of them is r7.1d-onestep-hs-fluid/config.py with the
metric_kwargs of its R6 counterpart; description and out_root follow the
name. Everything else as R7.1d: BO over x = (L_bridge, C_anchor, C_NaCl)
through the R3.2 eval-56 mapping file, R6 forward model, shift_rmse; box
mapping.physics_condition_box on the R6.8c ranges (r0 [1, 3.5], U0 [0.5, 8],
the R6.6 ranges), salt 0-40 mM, L_bridge >= 5 bp: L_bridge 5-183.15 bp
(r0 1.894-3.5), C_anchor 41.57-170.52; initial point (80 bp, 100, 20 mM);
batch BO 10 x 10 (max_evals 100, not R6.8c's 140), bo_seed 0, simulation
seed 42; GPU jobs on gpu-a40, 4 h each; S(q) on CPU both sides.

Submission: README.md next to this file.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mapping import load_mapping, physics_condition_box  # noqa: E402

TARGETS = HERE.parent / "inverse-targets"
MAPPING = REPO / "campaigns" / "r3.2-physics-validate" / "mapping" / "r3.2_eval056.json"

R6_R0 = (1.0, 3.5)                      # r6.8c-hs-lam2/config.py search (= R6.6)
R6_U0 = (0.5, 8.0)
SALT = (0.0, 40.0)                      # mM, as R7.1
L_BRIDGE_MIN = 5.0                      # bp (author, 2026-11-23)
INIT = {"L_bridge": 80.0, "C_anchor": 100.0, "C_NaCl": 20.0}

BOX = physics_condition_box(load_mapping(MAPPING), R6_R0, R6_U0, SALT, L_BRIDGE_MIN)

CONFIG = {
    "description": "R7.1f one-step, lambda_shift = 2: BO over (L_bridge, C_anchor, C_NaCl) through the R3.2 "
                   "eval-56 mapping; R6.6 target HS fluid phi=0.4 (CPU S(q)); salt 0-40 mM, "
                   "box = image inside the R6.6 r0 and U0 ranges, L_bridge >= 5 bp "
                   "(r0 >= 1.894); batch BO 10 x 10 on gpu-a40",

    # md5 37c39120c6d44871953769456f8f3b75 (cpu/cpu_info.json), as R6.6
    "target": str(TARGETS / "fluid_hs_rho0.763944_D23.21_pad4.309" / "cpu" / "target_sq.npy"),
    "N": 5000,                         # meta.json, as R6.6
    "density": 0.005,

    "mapping": str(MAPPING),
    "search": {name: {"bounds": BOX[name], "init": INIT[name]} for name in
               ("L_bridge", "C_anchor", "C_NaCl")},

    "max_evals": 100,                  # first batch included, resumes counted
    "batch_size": 10,
    "bo_seed": 0,

    "out_root": "/gscratch/zeelab/hanson/codes/DNA_lipid_silica/testing/inverse/"
                "r7/r7.1f-onestep-hs-lam2/Optimization_Results",
    "gp_log_dir": None,                # <out_root>/gp_log

    "potential": "mie_r0",
    "metric": "shift_rmse",
    "metric_kwargs": {"lambda_shift": 2.0},   # as R6.8c (r6.8c-hs-lam2/config.py)
    "scattering": "saxsfft",

    "mc_dfm_root": "/gscratch/zeelab/hanson/codes/DNA_lipid_silica/MC-DFM",

    # As R7.1 (one gpu-a40 job per evaluation, 4 h limit).
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

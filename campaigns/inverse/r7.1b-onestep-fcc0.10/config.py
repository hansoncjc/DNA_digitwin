"""
R7.1b: one-step inverse design (physics-based endpoint) against the R6.4
target, the FCC crystallite at rho* = 0.10 (author, 2026-11-23: the R7.1
setup on the other two R6 targets).

As R7.1 (r7.1-onestep-fcc0.05/config.py) except the target: BO over
x = (L_bridge, C_anchor, C_NaCl) through the R3.2 eval-56 mapping file, R6
forward model, shift_rmse (defaults, lambda = 1).

Search box: mapping.physics_condition_box, the R7.1 rule. For salt 0-40 mM,
the largest box in x whose image lies inside the R6.4 search ranges r0
[2, 3.5] and U0 [0.5, 8], with L_bridge >= 5 bp (author, 2026-11-23). The
R6.4 ranges equal R6.5's, so the box equals R7.1's (L_bridge 16.72-183.15 bp,
C_anchor 41.57-170.52); the 5 bp floor does not bind.
Initial point (80 bp, 100, 20 mM) as R7.1. Batch BO 10 x 10, bo_seed 0,
simulation seed 42; GPU jobs on gpu-a40, 4 h each; S(q) on CPU both sides.

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

R6_R0 = (2.0, 3.5)                      # r6.4-fcc0.10/config.py search
R6_U0 = (0.5, 8.0)
SALT = (0.0, 40.0)                      # mM, as R7.1
L_BRIDGE_MIN = 5.0                      # bp (author, 2026-11-23)
INIT = {"L_bridge": 80.0, "C_anchor": 100.0, "C_NaCl": 20.0}

BOX = physics_condition_box(load_mapping(MAPPING), R6_R0, R6_U0, SALT, L_BRIDGE_MIN)

CONFIG = {
    "description": "R7.1b one-step: BO over (L_bridge, C_anchor, C_NaCl) through the R3.2 "
                   "eval-56 mapping; R6.4 target FCC rho*=0.10 (CPU S(q)); salt 0-40 mM, "
                   "box = image inside the R6.4 r0 and U0 ranges, L_bridge >= 5 bp; "
                   "batch BO 10 x 10 on gpu-a40",

    # md5 9131767b29aa1004d8ffb092891ea359 (cpu/cpu_info.json), as R6.4
    "target": str(TARGETS / "fcc_rho0.10_D45.7_pad2.2" / "cpu" / "target_sq.npy"),
    "N": 5017,                         # meta.json, as R6.4
    "density": 0.005,

    "mapping": str(MAPPING),
    "search": {name: {"bounds": BOX[name], "init": INIT[name]} for name in
               ("L_bridge", "C_anchor", "C_NaCl")},

    "max_evals": 100,                  # first batch included, resumes counted
    "batch_size": 10,
    "bo_seed": 0,

    "out_root": "/gscratch/zeelab/hanson/codes/DNA_lipid_silica/testing/inverse/"
                "r7/r7.1b-onestep-fcc0.10/Optimization_Results",
    "gp_log_dir": None,                # <out_root>/gp_log

    "potential": "mie_r0",
    "metric": "shift_rmse",
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

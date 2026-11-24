"""
ON HOLD (author, 2026-11-23): do not submit. The R6.6 fluid target may be
replaced by a rescaled fluid whose spacing lies in the mapping's reach.

R7.1c: one-step inverse design (physics-based endpoint) against the R6.6
target, the hard-sphere fluid droplets at phi = 0.4 (author, 2026-11-23:
the R7.1 setup on the other two R6 targets; the result shows what the choice
of mapping does to a target outside its reach).

As R7.1 (r7.1-onestep-fcc0.05/config.py) except the target and the box:
BO over x = (L_bridge, C_anchor, C_NaCl) through the R3.2 eval-56 mapping
file, R6 forward model, shift_rmse (defaults, lambda = 1).

Search box: mapping.physics_condition_box, the R7.1 rule. For salt 0-40 mM,
the largest box in x whose image lies inside the R6.6 search ranges r0
[1, 3.5] and U0 [0.5, 8], with L_bridge >= 5 bp (author, 2026-11-23: a
bridge cannot be absent). With these coefficients r0 = 1 needs a negative
L_bridge, so the floor binds: L_bridge 5-183.15 bp, r0 1.894-3.5 (r0 is
1.849 even at L_bridge = 0). The fluid spacing needs r0 about 1.16 (R6.9:
q1 ratio x r0 = 1.160 near R6.6 eval 67), so the target spacing is out of
reach under this mapping; this run measures how close one-step gets.
C_anchor 41.57-170.52 and salt 0-40 mM as R7.1. n and m follow L_bridge
alone; at 5 bp n = 13.98 and m = 6.80, inside the R6.6 n range [5.5, 15].
Initial point (80 bp, 100, 20 mM) as R7.1. Batch BO 10 x 10 (max_evals 100,
not R6.6's 140), bo_seed 0, simulation seed 42; GPU jobs on gpu-a40, 4 h
each; S(q) on CPU both sides.

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

R6_R0 = (1.0, 3.5)                      # r6.6-hs-fluid/config.py search
R6_U0 = (0.5, 8.0)
SALT = (0.0, 40.0)                      # mM, as R7.1
L_BRIDGE_MIN = 5.0                      # bp (author, 2026-11-23)
INIT = {"L_bridge": 80.0, "C_anchor": 100.0, "C_NaCl": 20.0}

BOX = physics_condition_box(load_mapping(MAPPING), R6_R0, R6_U0, SALT, L_BRIDGE_MIN)

CONFIG = {
    "description": "R7.1c one-step: BO over (L_bridge, C_anchor, C_NaCl) through the R3.2 "
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
                "r7/r7.1c-onestep-hs-fluid/Optimization_Results",
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

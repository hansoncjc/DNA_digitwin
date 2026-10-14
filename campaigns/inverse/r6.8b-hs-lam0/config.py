"""
R6.8b (R6.8, loss hyperparameters): inverse design against the hard-sphere fluid droplets phi = 0.4
with lambda_shift = 0 (no spacing penalty; the curves are still aligned on x = q/q1, so the loss is M4 alone).

Everything else as R6.6 (r6.6-hs-fluid/config.py): target, N, density, search box, initial
point, batch BO 10 per batch, bo_seed 0, shift_rmse defaults apart from
lambda_shift, CPU S(q) on both sides. max_evals 140 (the same count as the
lambda = 1 run it is compared with). GPU jobs on gpu-a40.

    python inverse.py campaigns/inverse/r6.8b-hs-lam0/config.py --check
    sbatch -J inv_6.8b --time=<time> -o <run dir>/master_%j.out \
        --export=ALL,INV_REPO=$REPO,INV_CONFIG=$REPO/campaigns/inverse/r6.8b-hs-lam0/config.py \
        $REPO/campaigns/inverse/inverse-example/inverse_parallel.sbatch
"""
from pathlib import Path

TARGETS = Path(__file__).resolve().parents[1] / "inverse-targets"

CONFIG = {
    "description": "R6.8b hard-sphere fluid droplets phi = 0.4, lambda_shift = 0; CPU target S(q); batch BO 10 per batch",

    # md5 37c39120c6d44871953769456f8f3b75 (cpu/cpu_info.json)
    "target": str(TARGETS / "fluid_hs_rho0.763944_D23.21_pad4.309" / "cpu" / "target_sq.npy"),
    "N": 5000,                         # meta.json
    "density": 0.005,
    "search": {
        "U0":    {"bounds": (0.5, 8.0),  "init": 3.0},
        "r0":    {"bounds": (1.0, 3.5),  "init": 2.5},
        "n":     {"bounds": (5.5, 15.0), "init": 12.0},
        "m_rel": {"bounds": (0.0, 1.0),  "init": 2.0 / 7.0},   # m = 6 at n = 12
    },

    "max_evals": 140,                  # first batch included, resumes counted
    "batch_size": 10,
    "bo_seed": 0,

    "out_root": "/gscratch/zeelab/hanson/codes/DNA_lipid_silica/testing/inverse/"
                "r6/r6.8b-hs-lam0/Optimization_Results",
    "gp_log_dir": None,                # <out_root>/gp_log

    "potential": "mie_r0",
    "metric": "shift_rmse",
    "metric_kwargs": {"lambda_shift": 0.0},
    "scattering": "saxsfft",

    "mc_dfm_root": "/gscratch/zeelab/hanson/codes/DNA_lipid_silica/MC-DFM",

    # One gpu-a40 job per evaluation; time and mem as r6.4-fcc0.10/config.py.
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

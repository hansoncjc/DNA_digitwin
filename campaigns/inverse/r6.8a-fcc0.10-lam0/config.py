"""
R6.8a (R6.8, loss hyperparameters): inverse design against the FCC crystallite rho* = 0.10
with lambda_shift = 0 (no spacing penalty; the curves are still aligned on x = q/q1, so the loss is M4 alone).

Everything else as R6.4 (r6.4-fcc0.10/config.py): target, N, density, search box, initial
point, batch BO 10 per batch, bo_seed 0, shift_rmse defaults apart from
lambda_shift, CPU S(q) on both sides. max_evals 100 (the same count as the
lambda = 1 run it is compared with). GPU jobs on ckpt-g2.

    python inverse.py campaigns/inverse/r6.8a-fcc0.10-lam0/config.py --check
    sbatch -J inv_6.8a --time=<time> -o <run dir>/master_%j.out \
        --export=ALL,INV_REPO=$REPO,INV_CONFIG=$REPO/campaigns/inverse/r6.8a-fcc0.10-lam0/config.py \
        $REPO/campaigns/inverse/inverse-example/inverse_parallel.sbatch
"""
from pathlib import Path

TARGETS = Path(__file__).resolve().parents[1] / "inverse-targets"

CONFIG = {
    "description": "R6.8a FCC crystallite rho* = 0.10, lambda_shift = 0; CPU target S(q); batch BO 10 per batch",

    # md5 9131767b29aa1004d8ffb092891ea359 (cpu/cpu_info.json)
    "target": str(TARGETS / "fcc_rho0.10_D45.7_pad2.2" / "cpu" / "target_sq.npy"),
    "N": 5017,                         # meta.json
    "density": 0.005,
    "search": {
        "U0":    {"bounds": (0.5, 8.0),  "init": 3.0},
        "r0":    {"bounds": (2.0, 3.5),  "init": 2.5},
        "n":     {"bounds": (5.5, 15.0), "init": 12.0},
        "m_rel": {"bounds": (0.0, 1.0),  "init": 2.0 / 7.0},   # m = 6 at n = 12
    },

    "max_evals": 100,                  # first batch included, resumes counted
    "batch_size": 10,
    "bo_seed": 0,

    "out_root": "/gscratch/zeelab/hanson/codes/DNA_lipid_silica/testing/inverse/"
                "r6/r6.8a-fcc0.10-lam0/Optimization_Results",
    "gp_log_dir": None,                # <out_root>/gp_log

    "potential": "mie_r0",
    "metric": "shift_rmse",
    "metric_kwargs": {"lambda_shift": 0.0},
    "scattering": "saxsfft",

    "mc_dfm_root": "/gscratch/zeelab/hanson/codes/DNA_lipid_silica/MC-DFM",

    # One ckpt-g2 job per evaluation (preemptible; a preempted job is
    # resubmitted up to max_job_retries times); time and mem as
    # r6.5-fcc0.05/config.py.
    "parallel": True,
    "parallel_cfg": {
        "partition": "ckpt-g2",
        "account": "zeelab",
        "mem": "10G",
        "time": "05:00:00",
        "module_loads": ["cuda/11.8"],
        "venv_activate": "/gscratch/zeelab/hanson/codes/HOOMD-start/hoomd-venv/bin/activate",
        "max_job_retries": 2,
        "poll_interval": 30.0,
        "max_wait": 8 * 3600,          # from submission: 5 h job + 3 h in the queue
    },
}

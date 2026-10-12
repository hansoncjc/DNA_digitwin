"""
Example batch-BO configuration (batch_size > 1), for smoke tests and as a
format reference. Not an R6 campaign: max_evals is 12, and the target is the
old FCC 09 target (saxs-fft trim slice(3, -3), before the 2026-10-08 change).
Every key is as in config.py; the batch keys are commented here.

    python inverse.py campaigns/inverse/inverse-example/config_batch.py --check
    CFG=$REPO/campaigns/inverse/inverse-example/config_batch.py
    sbatch --export=ALL,INV_REPO=$REPO,INV_CONFIG=$CFG,INV_ARGS="--out OUT --steps 2000000" \
        $REPO/campaigns/inverse/inverse-example/inverse_parallel.sbatch
"""
CONFIG = {
    "description": "inverse.py batch smoke test (FCC 09 target, 3 batches of 4)",

    "target": "/gscratch/zeelab/hanson/codes/DNA_lipid_silica/testing/inverse/"
              "02_fcc/09_sim_param_candidate_metrics/target_sq.npy",
    "N": 5017,
    "density": 0.005,
    "search": {
        "U0":    {"bounds": (0.5, 8.0),  "init": 3.0},
        "r0":    {"bounds": (2.0, 3.5),  "init": 2.5},
        "n":     {"bounds": (5.5, 15.0), "init": 12.0},
        "m_rel": {"bounds": (0.0, 1.0),  "init": 2.0 / 7.0},   # m = 6 at n = 12
    },

    # Batch BO. Each batch evaluates min(batch_size, max_evals - successes)
    # points at once, one GPU job each. The first batch is the initial point
    # plus batch_size - 1 scrambled Sobol points (seeded by bo_seed); later
    # batches are qLogEI picks (optimize_acqf sequential=True) on a GP fitted
    # to every successful evaluation. max_evals counts successful evaluations
    # of any kind, first batch and resumes included; failed ones are filled
    # in by later batches. Here: batches of 4, 4, 4.
    "max_evals": 12,
    "batch_size": 4,                   # locked on a resume; > 1 needs parallel=True
    "bo_seed": 0,

    "out_root": "./inverse_batch_example_out",
    "gp_log_dir": None,                # None: <out_root>/gp_log (one row per batch)

    "potential": "mie_r0",
    "metric": "shift_rmse",
    "scattering": "saxsfft",

    "mc_dfm_root": "/gscratch/zeelab/hanson/codes/DNA_lipid_silica/MC-DFM",

    # batch_size > 1 needs parallel=True: the batch's GPU jobs run at the
    # same time (inverse_parallel.sbatch; the master runs on a CPU node).
    "parallel": True,
    "parallel_cfg": {
        "partition": "gpu-a40",
        "account": "zeelab",
        "mem": "10G",
        "time": "03:00:00",
        "module_loads": ["cuda/11.8"],
        "venv_activate": "/gscratch/zeelab/hanson/codes/HOOMD-start/hoomd-venv/bin/activate",
        "max_job_retries": 2,
        "poll_interval": 30.0,
        "max_wait": 6 * 3600,
    },
}

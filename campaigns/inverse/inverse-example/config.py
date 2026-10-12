"""
Example inverse-design configuration, for smoke tests and as a format
reference. Not an R6 campaign: max_evals is 3, and the target is the old
FCC 09 target (saxs-fft trim slice(3, -3), before the 2026-10-08 change).

Search box and initial point are the 2026-08 FCC 09 run's values
(_discrepancies.md §8.1), used here only as an example.

    python inverse.py campaigns/inverse/inverse-example/config.py --check
    python inverse.py campaigns/inverse/inverse-example/config.py --out OUT --steps 2000000
"""
CONFIG = {
    "description": "inverse.py smoke test (FCC 09 target, 3 evaluations)",

    # Target S(q): (N, 2) .npy [q (1/A), S]. Its md5 is recorded at the first
    # start; a resume refuses a target whose content changed.
    "target": "/gscratch/zeelab/hanson/codes/DNA_lipid_silica/testing/inverse/"
              "02_fcc/09_sim_param_candidate_metrics/target_sq.npy",

    # Forward model: run_simulation / convert_to_SAXS_fft defaults; only N here.
    "N": 5017,
    "density": 0.005,                  # fixed number density

    # Search box (bo.ParamSpace, mode="sim"). U0, r0, n and one of m / m_rel,
    # each {"bounds": (lo, hi), "init": x} or {"fixed": x}.
    # m = 4 + m_rel (n - 5): m_rel in [0, 1] keeps m in [4, n - 1].
    "search": {
        "U0":    {"bounds": (0.5, 8.0),  "init": 3.0},
        "r0":    {"bounds": (2.0, 3.5),  "init": 2.5},
        "n":     {"bounds": (5.5, 15.0), "init": 12.0},
        "m_rel": {"bounds": (0.0, 1.0),  "init": 2.0 / 7.0},   # m = 6 at n = 12
    },

    # Successful evaluations in total: first batch + acquisitions, resumes
    # included. Failed and abandoned evaluations do not count.
    "max_evals": 3,
    "bo_seed": 0,
    "batch_size": 1,                   # one candidate per BO step; config_batch.py for > 1

    "out_root": "./inverse_example_out",
    "gp_log_dir": None,                # None: <out_root>/gp_log

    # Selected by name; one value each for now.
    "potential": "mie_r0",
    "metric": "shift_rmse",            # defaults, compared over the curve overlap
    "scattering": "saxsfft",

    "mc_dfm_root": "/gscratch/zeelab/hanson/codes/DNA_lipid_silica/MC-DFM",

    # False: this process runs every simulation (inverse_sequential.sbatch).
    # True: one GPU job per evaluation, this process on a CPU node
    # (inverse_parallel.sbatch). See README "Inverse design (inverse.py)".
    "parallel": False,
    "parallel_cfg": {                  # used only when parallel is True
        "partition": "ckpt-g2",
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

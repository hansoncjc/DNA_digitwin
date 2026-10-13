"""
R6.6: inverse design against the hard-sphere fluid droplet target
(phi = 0.4, 16 frames; rerun of the 2026-09 HS fluid runs under the current
forward model and shift_rmse).

Batch BO: 100 successful evaluations in batches of 10 (first batch = initial
point + 9 Sobol points), GPU jobs on gpu-a40. Search box and initial point
as the 2026-09 HS fluid runs (_discrepancies.md §8.1: r0 in [1, 3.5]).

S(q) is computed on the CPU on both sides: the simulated S(q) by
convert_to_SAXS_fft (no device argument, C26 deferred) on the GPU job's
node, the target by cpu_targets.py on a Xeon Gold 6230R
(inverse-targets/README.md). CPU S(q) depends on the CPU model: another
cluster node gave shift_rmse 1.7e-3 against the FCC rho* = 0.10 CPU target
for the same configuration.

    python inverse.py campaigns/inverse/r6.6-hs-fluid/config.py --check
    sbatch -J inv_r6.6 --time=<min> -o <run dir>/master_%j.out \
        --export=ALL,INV_REPO=$REPO,INV_CONFIG=$REPO/campaigns/inverse/r6.6-hs-fluid/config.py \
        $REPO/campaigns/inverse/inverse-example/inverse_parallel.sbatch
"""
from pathlib import Path

TARGETS = Path(__file__).resolve().parents[1] / "inverse-targets"

CONFIG = {
    "description": "R6.6 HS fluid phi=0.4, CPU target S(q); batch BO 10 x 10. "
                   "S(q) on CPU both sides; CPU S(q) depends on the CPU model "
                   "(another node: shift_rmse 1.7e-3 vs the FCC 0.10 CPU target)",

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

    # 100 in the first start (2026-10-12, complete); raised to 140 for a
    # resume of 4 more batches before the 2026-10-13 maintenance.
    "max_evals": 140,                  # first batch included, resumes counted
    "batch_size": 10,
    "bo_seed": 0,

    "out_root": "/gscratch/zeelab/hanson/codes/DNA_lipid_silica/testing/inverse/"
                "r6/r6.6-hs-fluid/Optimization_Results",
    "gp_log_dir": None,                # <out_root>/gp_log

    "potential": "mie_r0",
    "metric": "shift_rmse",
    "scattering": "saxsfft",

    "mc_dfm_root": "/gscratch/zeelab/hanson/codes/DNA_lipid_silica/MC-DFM",

    # One gpu-a40 job per evaluation; time and mem as r6.4-fcc0.10/config.py
    # (rmax/r0 depends on U0, n, m only, so r0 down to 1 does not raise it).
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

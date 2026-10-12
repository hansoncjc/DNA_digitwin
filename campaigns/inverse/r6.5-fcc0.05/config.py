"""
R6.5: inverse design against the FCC crystallite at rho* = 0.05 (the R6.4
crystallite scaled to a second spacing).

Batch BO: 100 successful evaluations in batches of 10 (first batch = initial
point + 9 Sobol points). GPU jobs on ckpt-g2 (gpu-a40 holds R6.4 and R6.6).
Search box and initial point as R6.4; the r0 box is kept at [2, 3.5]
(user, 2026-10-12) although q1 * r0 ~ 0.0339 predicts r0 ~ 3.32 for this
target (meta.json).

S(q) is computed on the CPU on both sides: the simulated S(q) by
convert_to_SAXS_fft (no device argument, C26 deferred) on the GPU job's
node, the target by cpu_targets.py on a Xeon Gold 6230R
(inverse-targets/README.md). CPU S(q) depends on the CPU model: another
cluster node gave shift_rmse 1.7e-3 against the FCC rho* = 0.10 CPU target
for the same configuration. ckpt-g2 nodes are of several kinds.

    python inverse.py campaigns/inverse/r6.5-fcc0.05/config.py --check
    sbatch -J inv_r6.5 --time=<min> -o <run dir>/master_%j.out \
        --export=ALL,INV_REPO=$REPO,INV_CONFIG=$REPO/campaigns/inverse/r6.5-fcc0.05/config.py \
        $REPO/campaigns/inverse/inverse-example/inverse_parallel.sbatch
"""
from pathlib import Path

TARGETS = Path(__file__).resolve().parents[1] / "inverse-targets"

CONFIG = {
    "description": "R6.5 FCC rho*=0.05, CPU target S(q); batch BO 10 x 10 on ckpt-g2. "
                   "S(q) on CPU both sides; CPU S(q) depends on the CPU model "
                   "(another node: shift_rmse 1.7e-3 vs the FCC 0.10 CPU target)",

    # md5 780fd41a9f25b73dba663037a2415a25 (cpu/cpu_info.json)
    "target": str(TARGETS / "fcc_rho0.05_D57.6_pad1.738" / "cpu" / "target_sq.npy"),
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
                "r6/r6.5-fcc0.05/Optimization_Results",
    "gp_log_dir": None,                # <out_root>/gp_log

    "potential": "mie_r0",
    "metric": "shift_rmse",
    "scattering": "saxsfft",

    "mc_dfm_root": "/gscratch/zeelab/hanson/codes/DNA_lipid_silica/MC-DFM",

    # One ckpt-g2 job per evaluation (idle GPUs across UW, jobs <= 6 h,
    # preemptible; a preempted job is resubmitted up to max_job_retries
    # times). time: about 2.3 h at worst on an A40 (r6.4-fcc0.10/config.py),
    # with room for a slower card, under the 6 h limit. mem as R3.2 on
    # ckpt-g2 (10G).
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

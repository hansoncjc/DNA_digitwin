"""
R6.4: inverse design against the FCC crystallite at rho* = 0.10 (rerun of
the 2026-08 FCC 09 run under the current forward model and shift_rmse).

Batch BO: 100 successful evaluations in batches of 10 (first batch = initial
point + 9 Sobol points), GPU jobs on gpu-a40. Search box and initial point
as the FCC 09 run (_discrepancies.md §8.1).

S(q) is computed on the CPU on both sides: the simulated S(q) by
convert_to_SAXS_fft (no device argument, C26 deferred) on the GPU job's
node, the target by cpu_targets.py on a Xeon Gold 6230R
(inverse-targets/README.md). CPU S(q) depends on the CPU model: another
cluster node gave shift_rmse 1.7e-3 against this target for the same
configuration.

    python inverse.py campaigns/inverse/r6.4-fcc0.10/config.py --check
    sbatch -J inv_r6.4 --time=<min> -o <run dir>/master_%j.out \
        --export=ALL,INV_REPO=$REPO,INV_CONFIG=$REPO/campaigns/inverse/r6.4-fcc0.10/config.py \
        $REPO/campaigns/inverse/inverse-example/inverse_parallel.sbatch
"""
from pathlib import Path

TARGETS = Path(__file__).resolve().parents[1] / "inverse-targets"

CONFIG = {
    "description": "R6.4 FCC rho*=0.10, CPU target S(q); batch BO 10 x 10. "
                   "S(q) on CPU both sides; CPU S(q) depends on the CPU model "
                   "(another node: shift_rmse 1.7e-3 vs this target)",

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
                "r6/r6.4-fcc0.10/Optimization_Results",
    "gp_log_dir": None,                # <out_root>/gp_log

    "potential": "mie_r0",
    "metric": "shift_rmse",
    "scattering": "saxsfft",

    "mc_dfm_root": "/gscratch/zeelab/hanson/codes/DNA_lipid_silica/MC-DFM",

    # One gpu-a40 job per evaluation. time: R3.1 took 26.7-43.3 min per
    # condition on an A40 (about 15 min of it outside production), and time
    # follows the neighbour count; rmax/r0 reaches 4.14 in this box against
    # 2.52 for the slowest R3.1 condition, (4.14/2.52)^3 ~ 4.4, so about
    # 2.3 h at worst. mem: R3.1 and R3.2 run the same saxs-fft CPU path
    # (N_grid 600) in 10G.
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

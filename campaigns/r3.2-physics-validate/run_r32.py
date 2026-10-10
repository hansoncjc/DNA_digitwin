"""
R3.2 driver: BO over the physics-based mapping coefficients against the
ground truth in GT_ROOT, one GPU job per training condition per evaluation.

    python run_r32.py --gt GT_ROOT --out OUT_ROOT [--n-iters N]   # default train_config.N_ITERS
    python run_r32.py --gt GT_SMOKE --out OUT_SMOKE --smoke --steps 200000
    python run_r32.py --gt GT_ROOT --out OUT_TRUTH --truth      # one evaluation at the ground truth

Resumes from OUT_ROOT/bo_trajectory.csv when it exists (run_bo_resumable):
resubmit the same command after a walltime, maintenance or preemption. The
evaluation that was cut off is marked ABANDONED and never read, the next id is
past every existing eval_XXX folder, and the cap of N_ITERS + 1 successful
evaluations counts the earlier ones. ``bo.inspect_resume(OUT_ROOT)`` shows what
a restart would load without writing anything.

Loss: shift_rmse with its defaults. The flat hold at the low-q end of the
overlap was fixed on 2026-10-09; the first R3.2 attempt ran before the fix
and was discarded.

--truth evaluates the objective once at the R3.1 ground truth (BO seed 42,
ground truth seed 1); the per-condition loss is the stochastic floor used
to set the success threshold. It does not run BO.

Cluster settings (partition, account, venv, MC-DFM path) come from the
command line or the environment (defaults: the klone paths); see run_r32.sbatch.
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import train_config as config  # noqa: E402


def build_datasets(gt_root, ids=None):
    from datasets import Dataset, ExperimentalParams, SimulationParams

    datasets = []
    for c in config.TRAIN:
        if ids is not None and c["id"] not in ids:
            continue
        path = Path(gt_root) / f"{c['id']}.npy"
        if not path.exists():
            raise FileNotFoundError(f"ground truth missing: {path}")
        datasets.append(Dataset(
            id=c["id"],
            exp_path=path,
            exp=ExperimentalParams(L_bridge=c["L_bridge"], C_chol=c["C_chol"], C_NaCl=c["C_NaCl"]),
            sim=SimulationParams(density=config.DENSITY),
            datatype="sq",
        ))
    return datasets


def parallel_cfg(a):
    return {
        "partition": a.partition,
        "account": a.account,
        "gpus": 1,
        "mem": a.mem,
        "time": a.job_time,
        "module_loads": [m for m in a.modules.split(",") if m],
        "venv_activate": a.venv,
        "code_root": str(REPO),
        "mc_dfm_root": a.mc_dfm_root,
        "poll_interval": 30.0,
        "max_wait": a.max_wait_h * 3600,
        "max_job_retries": a.max_job_retries,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-iters", type=int, default=config.N_ITERS)
    ap.add_argument("--smoke", action="store_true", help="2 conditions, 1 iteration")
    ap.add_argument("--steps", type=int, default=None, help="smoke test only")
    ap.add_argument("--truth", action="store_true", help="one evaluation at the ground truth")
    ap.add_argument("--sequential", action="store_true", help="no Slurm jobs (local test)")
    ap.add_argument("--device", default=None, help="local test only (default: gpu)")
    # GPU jobs go to ckpt-g2 (idle GPUs across UW, jobs <= 6 h, preemptible):
    # zeelab's own GPU partitions (20 and 4 GPUs, not combinable) cannot run
    # the 24 jobs of one evaluation at once. The master runs on cpu-g2.
    ap.add_argument("--partition", default=os.environ.get("R32_GPU_PARTITION", "ckpt-g2"))
    ap.add_argument("--account", default=os.environ.get("R32_ACCOUNT", "zeelab"))
    ap.add_argument("--mem", default="10G")
    ap.add_argument("--job-time", default="03:00:00")
    ap.add_argument("--max-wait-h", type=float, default=6.0)
    ap.add_argument("--max-job-retries", type=int, default=2,
                    help="resubmissions of a failed or preempted GPU job before the evaluation fails")
    ap.add_argument("--modules", default=os.environ.get("R32_MODULES", "cuda/11.8"))
    ap.add_argument("--venv", default=os.environ.get("R32_VENV", ""))
    ap.add_argument("--mc-dfm-root", default=os.environ.get(
        "MC_DFM_ROOT", "/gscratch/zeelab/hanson/codes/DNA_lipid_silica/MC-DFM"))
    a = ap.parse_args()

    if a.mc_dfm_root:
        sys.path.insert(0, a.mc_dfm_root)
    import torch
    import bo

    ids = {"d0", "d23"} if a.smoke else None
    datasets = build_datasets(a.gt, ids)
    ps = bo.ParamSpace(config.PARAM_CFG)
    print(bo.describe_training_config(ps, mode="map"))

    sim_defaults = {"N": config.N, "plot": False}
    if a.device is not None:
        sim_defaults["device"] = a.device
    if a.steps is not None:
        if not a.smoke:
            ap.error("--steps is for --smoke only")
        sim_defaults["steps"] = int(a.steps)

    os.makedirs(a.out, exist_ok=True)
    ffpath = str(REPO / "formfactors" / "sasmodels_sphere_fit.txt")
    objective = bo.make_global_objective(
        datasets, ps, ffpath=ffpath, out_root=a.out, trim_tail=0,
        sim_defaults=sim_defaults, mode="map", scattering_method="saxsfft",
        scattering_kwargs={}, metric=config.METRIC, compare_q_range=None,
        metric_kwargs={}, parallel=not a.sequential, parallel_cfg=parallel_cfg(a),
    )

    if a.truth:
        x_phys = torch.tensor([config.GROUND_TRUTH[name] for name in ps._names],
                              dtype=torch.float64)
        loss = float(objective(ps.phys_to_unit(x_phys), ffpath=ffpath))
        print(f"[truth] total loss at the ground truth: {loss:.6f} "
              f"(per condition: {a.out}/loss_components.txt)")
        return

    n_iters = 1 if a.smoke else a.n_iters
    best, history = bo.run_bo_resumable(
        objective, ps, ffpath, a.out, n_iters=n_iters, seed=config.BO_SEED,
        gp_log_dir=os.path.join(a.out, "gp_log"),
    )
    print("[r32] best:", ps.decode(best), "loss:", min(history))


if __name__ == "__main__":
    main()

"""
R4 driver: BO over the physics-based mapping coefficients against six measured
SAXS curves, one GPU job per condition per evaluation (as R3.2).

    python run_r4.py --out OUT_ROOT --check-only   # startup checks, no BO
    python run_r4.py --out OUT_ROOT [--max-evals N]
    python run_r4.py --out OUT_SMOKE --smoke --steps 200000

The six I(q) files of ``r4_config.CONDITIONS`` are read from the repo copy
``r4_config.DATA_DIR`` (``data/``); ``--data`` points elsewhere for tests
only. Every start first runs the startup checks and stops on any failure:
  - each file exists and has the md5 in the config;
  - the measured S(q) (trim_tail, extract_exp_sq, as the objective computes
    it) has a first peak under the R4 shift_rmse settings and is not
    dispersed; q1, s_max - 1 and the q range are printed;
  - N_grid: a measured curve has no saxs-fft q grid, so there is nothing to
    match (bo.py checks saxs-fft targets in mode "sim" only); reported;
  - the initial point maps every condition inside check_mapped_params; the
    fraction of the search box that does is reported per condition.
The results go to OUT_ROOT/r4_startup_checks.json.

Resumes from OUT_ROOT/bo_trajectory.csv when it exists (run_bo_resumable):
resubmit the same command after a walltime, maintenance or preemption. The
evaluation that was cut off is marked ABANDONED, the next id is past every
existing eval_XXX folder, and MAX_EVALS counts the earlier successes.

Cluster settings come from the command line or the environment (defaults:
the klone paths); see run_r4.sbatch.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import r4_config as config  # noqa: E402

SMOKE_IDS = {"d0", "d5"}   # L_bridge 20 / C_anchor 60 and L_bridge 80


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head():
    """HEAD commit, with "-dirty" if tracked files differ from it."""
    try:
        head = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
            text=True).strip()
        return head + ("-dirty" if dirty else "")
    except Exception:
        return None


def conditions(ids=None):
    return [c for c in config.CONDITIONS if ids is None or c["id"] in ids]


def build_datasets(data_dir, ids=None):
    from datasets import Dataset, ExperimentalParams, SimulationParams

    return [
        Dataset(
            id=c["id"],
            exp_path=Path(data_dir) / c["file"],
            exp=ExperimentalParams(L_bridge=c["L_bridge"], C_anchor=c["C_anchor"],
                                   C_NaCl=c["C_NaCl"]),
            sim=SimulationParams(density=config.DENSITY),
            datatype=config.DATATYPE,
        )
        for c in conditions(ids)
    ]


def measured_sq(dataset, ffpath):
    """S(q) of one dataset exactly as the objective computes it."""
    from scattering import extract_exp_sq

    lo, hi = config.EXTRACT_Q_WINDOW
    return extract_exp_sq(dataset.load_exp_curve(trim_tail=config.TRIM_TAIL),
                          ffpath=ffpath, q_min=lo, q_max=hi, normalize=False)


def box_fraction(cond, n=4096, seed=0):
    """Fraction of a Sobol sample of the search box that maps ``cond`` inside the limits."""
    import torch
    from datasets import check_mapped_params, physics_map

    names = list(config.PARAM_CFG)
    lo = torch.tensor([config.PARAM_CFG[k]["bounds"][0] for k in names], dtype=torch.float64)
    hi = torch.tensor([config.PARAM_CFG[k]["bounds"][1] for k in names], dtype=torch.float64)
    X = lo + (hi - lo) * torch.quasirandom.SobolEngine(len(names), scramble=True, seed=seed).draw(n,
                                                                                               dtype=torch.float64)
    ok = 0
    for x in X:
        p = physics_map(cond["L_bridge"], cond["C_anchor"], cond["C_NaCl"],
                        **dict(zip(names, x.tolist())))
        try:
            check_mapped_params(p["U0"], p["n"], p["m"])
            ok += 1
        except ValueError:
            pass
    return ok / n


def startup_checks(datasets, ffpath, out_root):
    """Run the checks of the module docstring; raise SystemExit on any failure."""
    from datasets import check_mapped_params, physics_map
    from metrics import detect_first_peak, shift_rmse_params
    from scattering import estimate_saxsfft_n_grid

    p = shift_rmse_params(config.METRIC_KWARGS)
    init = {k: v["init"] for k, v in config.PARAM_CFG.items()}
    cond_by_id = {c["id"]: c for c in config.CONDITIONS}
    report = {"git_head": git_head(), "metric_kwargs": config.METRIC_KWARGS,
              "trim_tail": config.TRIM_TAIL, "extract_q_window": list(config.EXTRACT_Q_WINDOW),
              "initial_point": init, "conditions": []}
    problems = []
    print(f"[r4] git {report['git_head']}; shift_rmse {config.METRIC_KWARGS}; "
          f"trim_tail {config.TRIM_TAIL}; extraction window {config.EXTRACT_Q_WINDOW}")
    for ds in datasets:
        c = cond_by_id[ds.id]
        row = {"id": ds.id, "file": c["file"], "L_bridge": c["L_bridge"],
               "C_anchor": c["C_anchor"], "C_NaCl": c["C_NaCl"]}
        if not ds.exp_path.exists():
            problems.append(f"{ds.id}: missing {ds.exp_path}")
            report["conditions"].append(row)
            continue
        row["md5"] = md5(ds.exp_path)
        if row["md5"] != c["md5"]:
            problems.append(f"{ds.id}: md5 {row['md5']} != config {c['md5']}")
        sq = measured_sq(ds, ffpath)
        pk = detect_first_peak(sq, search_range=p["peak_search_range"],
                               prominence_frac=p["prominence_frac"],
                               dispersed_delta=p["dispersed_delta"])
        row.update({"n_points": int(sq.shape[0]), "q_min": float(sq[0, 0]),
                    "q_max": float(sq[-1, 0]), "q1": float(pk.q1), "s_max_minus_1": float(pk.s_max - 1),
                    "dispersed": bool(pk.dispersed), "n_peaks_found": int(pk.n_peaks_found),
                    "n_grid_estimate": estimate_saxsfft_n_grid(sq)})
        if pk.dispersed:
            problems.append(f"{ds.id}: measured S(q) is dispersed "
                            f"(s_max - 1 = {pk.s_max - 1:.3f} < {p['dispersed_delta']})")
        elif pk.n_peaks_found == 0:
            problems.append(f"{ds.id}: no first peak in {p['peak_search_range']}")
        m = physics_map(c["L_bridge"], c["C_anchor"], c["C_NaCl"], **init)
        row["init_mapped"] = m
        try:
            check_mapped_params(m["U0"], m["n"], m["m"])
        except ValueError as exc:
            problems.append(f"{ds.id}: initial point {exc}")
        row["box_fraction_inside_limits"] = box_fraction(c)
        report["conditions"].append(row)
        print(f"[r4] {ds.id} L {c['L_bridge']:.0f} C_anchor {c['C_anchor']:.0f} "
              f"NaCl {c['C_NaCl']:.0f}: "
              f"{row['n_points']} pts q {row['q_min']:.5f}-{row['q_max']:.5f}, "
              f"q1 {row['q1']:.5f}, s_max-1 {row['s_max_minus_1']:.3f}, "
              f"dispersed {row['dispersed']}; N_grid: "
              f"{'not a saxs-fft grid, no match needed' if row['n_grid_estimate'] is None else row['n_grid_estimate']}; "
              f"init U0 {m['U0']:.3f} r0 {m['r0']:.3f} n {m['n']:.3f} m {m['m']:.3f}; "
              f"box inside limits {100 * row['box_fraction_inside_limits']:.0f} %")
    report["problems"] = problems
    os.makedirs(out_root, exist_ok=True)
    Path(out_root, "r4_startup_checks.json").write_text(json.dumps(report, indent=2))
    if problems:
        raise SystemExit("[r4] startup checks failed:\n  " + "\n  ".join(problems))
    print("[r4] startup checks passed")


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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(config.DATA_DIR),
                    help="folder with the six I(q) files (default: the repo copy; tests only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-evals", type=int, default=config.MAX_EVALS,
                    help="successful evaluations in total, resumes included")
    ap.add_argument("--check-only", action="store_true", help="startup checks only")
    ap.add_argument("--smoke", action="store_true", help="2 conditions (d0, d5), 2 evaluations")
    ap.add_argument("--steps", type=int, default=None, help="smoke test only")
    ap.add_argument("--sequential", action="store_true", help="no Slurm jobs (local test)")
    ap.add_argument("--device", default=None, help="local test only (default: gpu)")
    ap.add_argument("--partition", default=os.environ.get("R4_GPU_PARTITION", "gpu-a40"))
    ap.add_argument("--account", default=os.environ.get("R4_ACCOUNT", "zeelab"))
    ap.add_argument("--mem", default="10G")
    ap.add_argument("--job-time", default="03:00:00")
    ap.add_argument("--max-wait-h", type=float, default=6.0)
    ap.add_argument("--max-job-retries", type=int, default=2,
                    help="resubmissions of a failed GPU job before the evaluation fails")
    ap.add_argument("--modules", default=os.environ.get("R4_MODULES", "cuda/11.8"))
    ap.add_argument("--venv", default=os.environ.get("R4_VENV", ""))
    ap.add_argument("--mc-dfm-root", default=os.environ.get(
        "MC_DFM_ROOT", "/gscratch/zeelab/hanson/codes/DNA_lipid_silica/MC-DFM"))
    a = ap.parse_args(argv)

    if a.mc_dfm_root:
        sys.path.insert(0, a.mc_dfm_root)
    import bo

    ids = SMOKE_IDS if a.smoke else None
    datasets = build_datasets(a.data, ids)
    ffpath = str(REPO / config.FORMFACTOR)
    startup_checks(datasets, ffpath, a.out)
    if a.check_only:
        return

    ps = bo.ParamSpace(config.PARAM_CFG)
    print(bo.describe_training_config(ps, mode="map"))
    sim_defaults = {"N": config.N, "plot": False}
    if a.device is not None:
        sim_defaults["device"] = a.device
    if a.steps is not None:
        if not a.smoke:
            ap.error("--steps is for --smoke only")
        sim_defaults["steps"] = int(a.steps)

    objective = bo.make_global_objective(
        datasets, ps, ffpath=ffpath, out_root=a.out, trim_tail=config.TRIM_TAIL,
        sim_defaults=sim_defaults, mode="map", scattering_method="saxsfft",
        scattering_kwargs={}, metric=config.METRIC, compare_q_range=None,
        metric_kwargs=dict(config.METRIC_KWARGS), parallel=not a.sequential,
        parallel_cfg=parallel_cfg(a),
    )
    max_evals = 2 if a.smoke else a.max_evals
    best, history = bo.run_bo_resumable(
        objective, ps, ffpath, a.out, max_evals=max_evals, seed=config.BO_SEED,
        gp_log_dir=os.path.join(a.out, "gp_log"),
    )
    print("[r4] best:", ps.decode(best), "loss:", min(history))


if __name__ == "__main__":
    main()

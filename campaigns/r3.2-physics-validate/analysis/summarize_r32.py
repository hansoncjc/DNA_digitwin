"""
Tables for the R3.2 note (2026-10-09 BO_new_mapping_all_coeff), Results 1-4.

Usage: python summarize_r32.py RUN_DIR GT_DIR

RUN_DIR is the downloaded Optimization_Results copy (bo_trajectory.csv,
loss_components.txt, the master log r32_master_*.out, gp_log/, and whichever
eval_XXX folders were downloaded); GT_DIR is the R3.1 Sweep8_R3.1_GT folder.
Both are read only. Writes next to this script:
  r32_evals.csv       one row per evaluation: coefficients, a_n, total loss,
                      best so far, launcher time, GPU-job failures/retries,
                      requeues (pending count going up), d2 state, repeat set
  r32_conditions.csv  one row per training condition: mapped (U0, r0, n, m)
                      at the ground truth and at the best evaluation, loss at
                      the best evaluation, loss over the repeat set, first-peak
                      heights from the downloaded evaluations
  r32_gpu.csv         GPU model per host, from the job logs of the downloaded
                      evaluations
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]   # repo root
sys.path.insert(0, str(REPO))
from datasets import physics_map  # noqa: E402

COEFFS = ["k", "A", "K_s", "a_m", "delta"]
LO = np.array([0.40, 0.85, 0.00, 2.91, 1.09])     # train_config.PARAM_CFG
HI = np.array([0.90, 2.50, 0.10, 4.00, 4.00])
GT = {"k": 0.65, "A": 1.2, "K_s": 0.07, "a_m": 3.2, "delta": 3.2}
REPEAT_RADIUS = 0.05   # unit-cube distance to the last evaluation that defines the repeat set


def read_trajectory(path):
    lines = Path(path).read_text().splitlines()
    header = next(l for l in lines if l.startswith("iteration,"))
    cols = header.split(",")
    rows = [l.split(",") for l in lines if l and not l.startswith("#") and not l.startswith("iteration,")]
    df = pd.DataFrame(rows, columns=cols)
    for c in cols:
        if c != "dataset_id":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["iteration"] = df["iteration"].astype(int)
    return df


def read_master(path):
    """Per evaluation: launcher time, job summaries, retries, requeues."""
    ev = {}
    cur = None
    prev = None
    for line in Path(path).read_text().splitlines():
        m = re.match(r"\[launcher\] run_dir = \S*eval_(\d+)/_jobs(/retry_\d+)?", line)
        if m:
            cur = int(m.group(1))
            d = ev.setdefault(cur, {"launcher_s": 0.0, "jobs": {}, "retries": 0, "requeues": 0})
            if m.group(2):
                d["retries"] += 1
            d["_t"] = 0.0
            prev = None
            continue
        m = re.match(r"\[\s*([\d.]+)s\] done=(\d+) metric_failed=(\d+) failed=(\d+) running=(\d+) pending=(\d+)", line)
        if m and cur is not None:
            t, done, mfail, fail, run, pend = float(m.group(1)), *map(int, m.groups()[1:])
            d = ev[cur]
            d["launcher_s"] += t - d["_t"]
            d["_t"] = t
            if prev is not None and pend > prev:
                d["requeues"] += pend - prev
            d["metric_failed_max"] = max(d.get("metric_failed_max", 0), mfail)
            prev = pend
            continue
        m = re.match(r"\s+(i\d+_d\d+(?:_r\d+)?)\s+slurm=(\d+)\s+status=(\S+)\s+host=(\S+)\s+run time=(\S+)", line)
        if m and cur is not None:
            name, sid, status, host, rt = m.groups()
            ev[cur]["jobs"][name] = {"slurm": sid, "status": status, "host": host,
                                     "run_time_s": None if rt == "n/a" else float(rt.rstrip("s"))}
    return ev


def gpu_by_host(run):
    out = {}
    for f in glob.glob(str(Path(run) / "eval_*" / "_jobs" / "jobs" / "*.out")):
        txt = Path(f).read_text(errors="replace").splitlines()
        node = next((l.split()[-1] for l in txt if l.startswith("Node:")), None)
        for i, l in enumerate(txt):
            if l.startswith("name, memory.total") and i + 1 < len(txt):
                out.setdefault(node, set()).add(txt[i + 1].split(",")[0].strip())
    return {h: "/".join(sorted(v)) for h, v in out.items()}


def main(run, gt_dir):
    run, gt_dir, out = Path(run), Path(gt_dir), Path(__file__).resolve().parent
    out.mkdir(parents=True, exist_ok=True)
    traj = read_trajectory(run / "bo_trajectory.csv")
    lc = pd.read_csv(run / "loss_components.txt")
    tot = traj.groupby("iteration").loss.sum()
    assert np.allclose(tot.values, lc.groupby("iteration").loss.sum().reindex(tot.index).values)

    # ------------------------------------------------------------ per evaluation
    ev = traj.groupby("iteration")[COEFFS].first()
    ev["a_n"] = ev.a_m + ev.delta
    ev["loss"] = tot
    ev["best_so_far"] = tot.cummin()
    unit = (ev[COEFFS].values - LO) / (HI - LO)
    last = unit[-1]
    ev["dist_to_last"] = np.sqrt(((unit - last) ** 2).sum(1))
    ev["repeat_set"] = ev.dist_to_last < REPEAT_RADIUS
    sh = traj.pivot(index="iteration", columns="dataset_id", values="shift")
    ev["d2_sim_dispersed"] = sh["d2"] == 0.0          # target d2 is peaked: shift 0 <=> sim dispersed
    ev["n_peaked_sim_dispersed"] = sum(
        (sh[c] == 0.0).astype(int) for c in sh.columns if (sh[c] != 0.0).any())
    master_files = sorted(run.glob("r32_master_*.out"))
    if master_files:
        ms = read_master(master_files[0])
        for e, d in ms.items():
            if e not in ev.index:     # cut off by the walltime: no trajectory block
                print(f"eval {e}: in the master log, not in the trajectory; "
                      f"{len(d['jobs'])} job summaries, launcher ran {d['launcher_s'] / 60:.1f} min")
                continue
            rts = [j["run_time_s"] for j in d["jobs"].values() if j["status"] == "DONE" and j["run_time_s"]]
            ev.loc[e, "launcher_s"] = d["launcher_s"]
            ev.loc[e, "n_job_records"] = len(d["jobs"])
            ev.loc[e, "failed_jobs"] = sum(j["status"] != "DONE" for j in d["jobs"].values())
            ev.loc[e, "retries"] = d["retries"]
            ev.loc[e, "requeues"] = d["requeues"]
            ev.loc[e, "metric_failed_max"] = d.get("metric_failed_max", 0)
            ev.loc[e, "job_rt_median_s"] = np.median(rts) if rts else np.nan
            ev.loc[e, "job_rt_max_s"] = max(rts) if rts else np.nan
        jobs = pd.DataFrame([dict(eval=e, name=n, **j) for e, d in ms.items() for n, j in d["jobs"].items()])
    ev.index.name = "eval"
    ev.to_csv(out / "r32_evals.csv", float_format="%.6g")

    # ------------------------------------------------------------ per condition
    best = int(tot.idxmin())
    rep = list(ev.index[ev.repeat_set])
    piv = lc.pivot(index="iteration", columns="dataset_id", values="loss")
    rows = []
    for cid in sorted(lc.dataset_id.unique(), key=lambda s: int(s[1:])):
        g = json.loads((gt_dir / "ground_truth" / cid / "gt_params.json").read_text())
        c = g["condition"]   # key "C_chol" as written in gt_params.json at 49aac10
        mg = physics_map(c["L_bridge"], c["C_chol"], c["C_NaCl"], **GT)
        mb = physics_map(c["L_bridge"], c["C_chol"], c["C_NaCl"], **ev.loc[best, COEFFS].to_dict())
        tb = traj[(traj.iteration == best) & (traj.dataset_id == cid)].iloc[0]
        assert np.isclose(tb.U0, mb["U0"]) and np.isclose(tb.r0, mb["r0"]) and np.isclose(tb.n, mb["n"])
        r = {"id": cid, "L_bridge": c["L_bridge"], "C_anchor": c["C_chol"], "C_NaCl": c["C_NaCl"]}
        for key in ("U0", "r0", "n", "m"):
            r[f"{key}_gt"] = mg[key]
            r[f"{key}_best"] = mb[key]
        r["loss_best"] = piv.loc[best, cid]
        r["rmse_best"] = tb.rmse
        r["shift_best"] = tb["shift"]
        r["rep_mean"] = piv.loc[rep, cid].mean()
        r["rep_sd"] = piv.loc[rep, cid].std(ddof=1)
        r["rep_min"] = piv.loc[rep, cid].min()
        r["rep_max"] = piv.loc[rep, cid].max()
        for e in sorted(int(p.name[5:]) for p in run.glob("eval_*") if p.is_dir()):
            f = run / f"eval_{e:03d}" / cid / "shift_rmse_diagnostics.json"
            if f.is_file():
                d = json.loads(f.read_text())
                r["smax1_tgt"] = d["peak_tgt"]["s_max"] - 1
                r[f"smax1_sim_e{e}"] = d["peak_sim"]["s_max"] - 1
                r[f"loss_e{e}"] = d["loss"]
                assert np.isclose(d["loss"], piv.loc[e, cid])
        rows.append(r)
    cond = pd.DataFrame(rows)
    cond.to_csv(out / "r32_conditions.csv", index=False, float_format="%.6g")

    # ------------------------------------------------------------ GPU per host
    gh = gpu_by_host(run)
    gpu = pd.DataFrame(sorted(gh.items()), columns=["host", "gpu"])
    if master_files:
        cnt = jobs.groupby("host").size()
        gpu["jobs_in_run"] = gpu.host.map(cnt).fillna(0).astype(int)
        jobs["gpu"] = jobs.host.map(gh)
    gpu.to_csv(out / "r32_gpu.csv", index=False)

    # ------------------------------------------------------------ printout for the note
    n = len(ev)
    print(f"evaluations in the trajectory: {n} (ids {ev.index.min()}-{ev.index.max()})")
    print(f"best: eval {best}, total {tot[best]:.4f}")
    for c in COEFFS + ["a_n"]:
        g = GT[c] if c != "a_n" else GT["a_m"] + GT["delta"]
        print(f"  {c:6s} best {ev.loc[best, c]:.4f}  gt {g:.4f}  rel {100 * (ev.loc[best, c] - g) / g:+.2f} %")
    print(f"repeat set ({len(rep)} evals, unit distance < {REPEAT_RADIUS} to eval {ev.index[-1]}): {rep}")
    rs = ev.loc[rep]
    print(f"  total loss mean {rs.loss.mean():.4f} sd {rs.loss.std(ddof=1):.4f} min {rs.loss.min():.4f} max {rs.loss.max():.4f}")
    for c in COEFFS + ["a_n"]:
        g = GT[c] if c != "a_n" else GT["a_m"] + GT["delta"]
        print(f"  {c:6s} mean {rs[c].mean():.4f} range {rs[c].min():.4f}-{rs[c].max():.4f}  rel {100 * (rs[c].mean() - g) / g:+.2f} %")
    first = {thr: int(ev.index[ev.loss < thr][0]) for thr in (1.0, 0.8, 0.75, 0.7, 0.65) if (ev.loss < thr).any()}
    print("first evaluation below:", first)
    print("d2 simulated dispersed at evals:", list(ev.index[ev.d2_sim_dispersed]))
    if master_files:
        done = jobs[jobs.status == "DONE"]
        print(f"GPU job records {len(jobs)}, DONE {len(done)}, not DONE {len(jobs) - len(done)}: "
              f"{jobs[jobs.status != 'DONE'][['eval', 'name', 'status', 'host']].values.tolist()}")
        print(f"retries {int(ev.retries.sum())}; requeues (pending count rising) {int(ev.requeues.sum())} "
              f"in evals {list(ev.index[ev.requeues > 0])}; METRIC_FAILED max {int(ev.metric_failed_max.max())}")
        lt = ev.launcher_s.dropna()
        print(f"launcher time per evaluation: median {lt.median() / 60:.1f} min, range {lt.min() / 60:.1f}-{lt.max() / 60:.1f} min, "
              f"sum {lt.sum() / 3600:.1f} h; longest evals {lt.sort_values().index[-4:].tolist()}")
        rt = done.run_time_s
        print(f"GPU job run time: median {rt.median() / 60:.1f} min, range {rt.min() / 60:.1f}-{rt.max() / 60:.1f} min over {len(rt)} jobs")
        print(f"hosts {jobs.host.nunique() - ('-' in set(jobs.host))}; GPU model known for {jobs.gpu.notna().sum()} of {len(jobs)} job records")
        print(done.groupby(done.gpu.fillna("unknown")).run_time_s.describe()[["count", "50%", "min", "max"]].round(0))
    print("audit: max n_pairs_below_rmin", int(traj.n_pairs_below_rmin.max()),
          "max rmax", round(traj.rmax.max(), 2), "max rmax/r0", round(traj.rmax_over_r0.max(), 3))
    print(gpu.to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:3])

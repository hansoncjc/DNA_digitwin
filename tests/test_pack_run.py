"""campaigns/inverse/analysis/pack_run.py on a fake batch-BO run written with bo's own writers."""
import csv
import json
import os
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

import bo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "campaigns" / "inverse" / "analysis"))
import pack_run  # noqa: E402

SEARCH = {
    "U0":    {"bounds": [0.5, 8.0],  "init": 3.0},
    "r0":    {"bounds": [1.0, 3.5],  "init": 2.5},
    "n":     {"bounds": [5.5, 15.0], "init": 12.0},
    "m_rel": {"bounds": [0.0, 1.0],  "init": 2.0 / 7.0},
}
T0 = time.mktime(datetime(2026, 10, 12, 2, 0, 0).timetuple())


def _date(epoch, ampm=False):
    fmt = "%a %b %d %I:%M:%S %p PDT %Y" if ampm else "%a %b %d %H:%M:%S PDT %Y"
    return datetime.fromtimestamp(epoch).strftime(fmt)


def _params(eval_id):
    if eval_id == 0:
        return {k: v["init"] for k, v in SEARCH.items()}
    rng = np.random.default_rng(eval_id)
    return {k: float(lo + (hi - lo) * rng.random()) for k, (lo, hi) in
            ((k, v["bounds"]) for k, v in SEARCH.items())}


def _record(eval_id, p, loss, rmse, shift):
    m = 4 + p["m_rel"] * (p["n"] - 5)
    return {"iteration": eval_id, "dataset_id": "d0", "loss": loss,
            "k": "", "A": "", "K_s": "", "a_m": "", "delta": "",
            "density": 0.005, "n": p["n"], "m": m, "r0": p["r0"], "U0": p["U0"],
            "m_rel": p["m_rel"], "rmax": 2.0 * p["r0"], "rmax_over_r0": 2.0 + p["U0"] / 8,
            "t_tol": 0.1 / p["U0"], "n_pairs_below_rmin": 0, "rmse": rmse, "shift": shift}


def _write_job(eval_dir, name, slurm_id, submit, start, end, status, sub="", ampm=False):
    run_dir = eval_dir / "_jobs" / sub if sub else eval_dir / "_jobs"
    (run_dir / "jobs").mkdir(parents=True, exist_ok=True)
    sbatch = run_dir / "jobs" / f"{name}.sbatch"
    sbatch.write_text("#!/bin/bash\n")
    os.utime(sbatch, (submit, submit))
    (run_dir / "jobs" / f"{name}.out").write_text(
        f"Job:       {name}\nJobID:     {slurm_id}\nNode:      g3001\n"
        f"Start:     {_date(start, ampm)}\n"
        "name, memory.total [MiB]\nNVIDIA A40, 46068 MiB\n"
        "Average TPS: 9000.5\nAverage TPS: 4321.0\n"
        f"End:       {_date(end, ampm)}\n")
    (run_dir / "summary.json").write_text(json.dumps({"jobs": [{
        "name": name, "ds_id": "d0", "slurm_id": slurm_id, "status": status,
        "host": "g3001", "run_time_seconds": end - start - 30, "sim_dir": ""}]}))


def _write_eval(out, eval_id, loss=None, status="DONE", dispersed=False, hours=1.0,
                queue=600.0, retry=False):
    """eval_XXX/d0 with the files of a parallel evaluation; times from T0."""
    p = _params(eval_id)
    ds = out / f"eval_{eval_id:03d}" / "d0"
    (ds / "S(q)_data").mkdir(parents=True)
    with open(ds / "sim_params_d0.csv", "w", newline="") as fh:
        rec = {k: v for k, v in _record(eval_id, p, "", "", "").items()
               if k in bo._TRAJECTORY_PARAM_COLUMNS}
        w = csv.DictWriter(fh, fieldnames=list(rec))
        w.writeheader()
        w.writerow(rec)
    submit = T0 + 3600 * eval_id
    start = submit + queue
    end = start + hours * 3600
    if retry:
        _write_job(ds.parent, f"i{eval_id}_d00", str(1000 + eval_id), submit, start,
                   start + 60, "FAILED")
        submit_r = start + 120
        start = submit_r + queue
        end = start + hours * 3600
        _write_job(ds.parent, f"i{eval_id}_d00_r1", str(2000 + eval_id), submit_r, start,
                   end, status, sub="retry_1", ampm=True)
    else:
        _write_job(ds.parent, f"i{eval_id}_d00", str(1000 + eval_id), submit, start, end,
                   status)
    worker_end = end - 10
    worker_s = worker_end - (start + 20)
    gsd = ds / "DNA_assembly_20261012020000.gsd"
    gsd.write_bytes(b"gsd" * 100)
    os.utime(gsd, (worker_end - 300, worker_end - 300))
    npy = ds / "S(q)_data" / "average_structure_factor.npy"
    np.save(npy, np.column_stack([np.linspace(0.002, 0.13, 50), np.ones(50)]))
    os.utime(npy, (worker_end - 60, worker_end - 60))
    (ds / "S(q)_data" / "average_structure_factor.png").write_bytes(b"png")
    (ds / "potential_energy.csv").write_text("t,E\n")
    if status == "DONE":
        flag = json.dumps({"run_time_seconds": worker_s, "host": "g3001", "loss": loss})
        diag = {"metric": "shift_rmse", "failed": False, "loss": loss,
                "m4": loss * 0.9, "shift_term": loss * 0.1,
                "q1_tgt": 0.02717, "q1_sim": None if dispersed else 0.02717 * 1.01,
                "q1_ratio": None if dispersed else 1.01, "aligned": not dispersed,
                "dispersed_tgt": False, "dispersed_sim": dispersed, "anchor_scale": 1.0}
        (ds / "shift_rmse_diagnostics.json").write_text(json.dumps(diag))
    elif status == "METRIC_FAILED":
        flag = json.dumps({"run_time_seconds": worker_s, "host": "g3001",
                           "reason": "no simulated peak"})
        (ds / "shift_rmse_diagnostics.json").write_text(json.dumps(
            {"metric": "shift_rmse", "failed": True, "reason": "no simulated peak"}))
    else:
        flag = f"run_time_seconds={worker_s:.1f}\nhost=g3001\n\nTraceback: boom\n"
    (ds / status).write_text(flag)
    os.utime(ds / status, (worker_end, worker_end))
    return p


@pytest.fixture
def run(tmp_path):
    """22 evaluations: batch 0 (0-9), GP batch 1 (10-19), GP batch 2 (20-22, 22 still running)."""
    out = tmp_path / "r6.6-hs-fluid" / "Optimization_Results"
    out.mkdir(parents=True)
    traj = str(out / "bo_trajectory.csv")
    (out / "inverse_config.json").write_text(json.dumps({
        "description": "fake R6.6", "target": {"path": "target_sq.npy", "md5": "abc"},
        "locked": {"search": SEARCH, "batch_size": 10, "gp_log_dir": "gp_log"},
        "starts": [{"time": "2026-10-12T02:00:00", "commit": "7eba805", "dirty": False,
                    "host": "n1", "slurm_job_id": "41773154", "max_evals": 100}]}))
    losses = {i: 1.0 - 0.02 * i for i in range(22)}
    losses[4] = 0.30          # best of batch 0
    losses[5] = 0.95          # dispersed
    for eval_id in range(22):
        if eval_id == 13:
            _write_eval(out, eval_id, status="FAILED")
            bo._write_failed_iteration_block(traj, eval_id, [], reason="FAILED")
            continue
        if eval_id == 15:
            _write_eval(out, eval_id, status="METRIC_FAILED")
            bo._write_failed_iteration_block(traj, eval_id, [], reason="METRIC_FAILED",
                                             status="METRIC_FAILED")
            continue
        loss = losses[eval_id]
        p = _write_eval(out, eval_id, loss=loss, dispersed=(eval_id == 5),
                        hours=1.9 if eval_id == 7 else 1.0,
                        queue=5400.0 if eval_id == 18 else 600.0,
                        retry=(eval_id == 17))
        rec = _record(eval_id, p, loss, loss * 0.9, loss * 0.1)
        bo._write_iteration_block(traj, eval_id, loss, [rec])
        bo._append_loss_components(str(out), eval_id, loss, [rec])
    (out / "eval_022" / "d0").mkdir(parents=True)
    (out / "eval_022" / "d0" / "RUNNING").write_text("pid=1\n")

    gp = out / "gp_log"
    (gp / "states").mkdir(parents=True)
    rows = []
    for batch, ids in ((1, list(range(10, 20))), (2, [20, 21, 22])):
        state = f"states/ntrain_{batch}_attempt_0.pt"
        (gp / state).write_bytes(b"pt")
        rows.append({"gp_log_version": 2, "batch": batch, "n_train": 10 * batch,
                     "stage": "acquisition", "attempt": 0,
                     "param_names": ["U0", "r0", "n", "m_rel"],
                     "lengthscale": [0.5 * batch, 0.25, 1.0, 2.0], "outputscale": 1.0,
                     "noise": 1e-4, "eval_ids": ids, "best_observed_loss": 0.3,
                     "state_file": state})
    (gp / "gp_log.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n" + '{"gp_log_version": 2, "bat')
    return out


def test_summary_numbers(run):
    s = pack_run.collect(run, top_k=3, use_sacct=False)
    by_id = {r["eval_id"]: r for r in s["evals"]}

    assert [r["eval_id"] for r in s["top"]] == [4, 21, 20]
    assert s["top"][0]["loss"] == pytest.approx(0.30)
    assert s["top"][0]["m"] == pytest.approx(4 + by_id[4]["m_rel"] * (by_id[4]["n"] - 5))
    assert s["top"][1]["q1_ratio"] == pytest.approx(1.01)
    assert s["top"][0]["gsd_md5"] is not None

    c = s["counts"]
    assert (c["ok"], c["FAILED"], c["METRIC_FAILED"], c["no_block"]) == (20, 1, 1, 1)
    assert c["dispersed_sim_ok"] == 1 and c["not_aligned_ok"] == 1 and c["retried"] == 1

    assert by_id[0]["origin"] == "init"
    assert {by_id[i]["origin"] for i in range(1, 10)} == {"sobol"}
    assert by_id[12]["batch"] == 1 and by_id[12]["origin"] == "gp"
    assert by_id[22]["status"] == "no_block" and by_id[22]["batch"] == 2
    assert by_id[13]["n"] == pytest.approx(_params(13)["n"])      # from sim_params
    assert by_id[13]["reason"] == "Traceback: boom"         # last line of the FAILED flag

    b = {x["batch"]: x for x in s["batches"]}
    assert b[0]["best_eval"] == 4 and b[0]["cum_best"] == pytest.approx(0.30)
    assert b[1]["n_failed"] == 2 and b[1]["cum_best"] == pytest.approx(0.30)
    assert s["first_batch"]["top_k_from_first_batch"] == [4]
    assert s["first_batch"]["n_later_better_than_first_best"] == 0

    assert by_id[3]["queue_s"] == pytest.approx(600.0)
    assert by_id[3]["total_s"] == pytest.approx(600.0 + 3600.0)
    assert by_id[3]["sim_s"] == pytest.approx(by_id[3]["worker_s"] - 300.0)
    assert by_id[3]["sq_s"] == pytest.approx(240.0)
    assert by_id[3]["tps"] == pytest.approx(4321.0) and by_id[3]["gpu"] == "NVIDIA A40"
    assert by_id[17]["n_attempts"] == 2 and by_id[17]["queue_s"] == pytest.approx(1200.0)

    slow = {r["eval_id"]: r for r in pack_run.slow_evals(s, 1.5)}
    assert set(slow) == {7, 18}                     # 17: 1.37 h over two attempts
    assert slow[7]["dominant"] == "simulation"
    assert slow[18]["dominant"] == "queue"          # 1.5 h queue + 1 h job

    ls = {(r["batch"], r["stage"]): r for r in s["gp_lengthscale"]}
    assert len(s["gp_lengthscale"]) == 2            # the cut last line is skipped
    assert ls[(2, "acquisition")]["ls_U0_phys"] == pytest.approx(1.0 * 7.5)


def test_pack_tar(run, tmp_path):
    dest = tmp_path / "dest"
    assert pack_run.main([str(run), "--top", "3", "--name", "r6.6", "--dest", str(dest),
                          "--no-sacct"]) == 0
    with tarfile.open(dest / "r6.6_pack.tar.gz") as tar:
        names = set(tar.getnames())
    root = "r6.6/Optimization_Results/"
    for name in ("bo_trajectory.csv", "loss_components.txt", "inverse_config.json",
                 "gp_log/gp_log.jsonl", "gp_log/states/ntrain_1_attempt_0.pt",
                 "eval_005/d0/S(q)_data/average_structure_factor.npy",
                 "eval_005/d0/shift_rmse_diagnostics.json", "eval_005/d0/sim_params_d0.csv",
                 "eval_013/d0/FAILED", "eval_017/_jobs/retry_1/summary.json",
                 "eval_004/d0/DNA_assembly_20261012020000.gsd",
                 "eval_004/d0/S(q)_data/average_structure_factor.png"):
        assert root + name in names, name
    gsds = sorted(n for n in names if n.endswith(".gsd"))
    assert gsds == [f"{root}eval_{i:03d}/d0/DNA_assembly_20261012020000.gsd" for i in (4, 20, 21)]
    for name in ("summary.json", "summary.txt", "evals.csv", "batches.csv",
                 "gp_lengthscale.csv", "manifest.txt"):
        assert f"r6.6/pack_summary/{name}" in names
    # out_root is read only
    assert not any(p.name.endswith((".tar.gz", "_pack_summary")) for p in run.iterdir())
    text = (dest / "r6.6_pack_summary" / "summary.txt").read_text()
    assert "evaluations with total >= 1.5 h: 2" in text

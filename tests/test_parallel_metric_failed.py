"""METRIC_FAILED in the parallel path: worker flag, no resubmission, failed eval."""
import json
from pathlib import Path

import numpy as np
import pytest

import bo
from parallel import submit_parallel as sp
from parallel import worker
from shift_rmse_curves import crystal_curve, flat_curve


def _worker_cfg(tmp_path, metric_kwargs):
    exp_path = tmp_path / "target.npy"
    np.save(exp_path, crystal_curve())
    outdir = tmp_path / "eval_000" / "d0"
    cfg = {
        "outdir": str(outdir),
        "run_kwargs": {"density": 0.005, "U_0": 3.0, "r0": 2.5, "n": 12.0, "m": 6.0},
        "scattering": {"method": "saxsfft", "kwargs": {}},
        "loss": {
            "exp_path": str(exp_path), "trim_tail": 0, "datatype": "sq",
            "ffpath": "", "metric": "shift_rmse", "scattering_method": "saxsfft",
            "compare_q_range": [0.004, 0.040], "dp_coeff": 0.5, "plot_apdist": True,
            "metric_kwargs": metric_kwargs,
        },
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    return cfg_path, outdir


@pytest.fixture
def fake_pipeline(monkeypatch):
    def fake_sim(outdir, **kw):
        return {"rmax": 4.9465, "t_tol_lj": 0.1 / kw["U_0"], "n_pairs_below_rmin": 2}

    def fake_saxs(save_dir, **kw):
        d = Path(save_dir) / "S(q)_data"
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "average_structure_factor.npy", fake_saxs.curve)

    monkeypatch.setattr(worker, "run_simulation", fake_sim)
    monkeypatch.setattr(worker, "convert_to_SAXS_fft", fake_saxs)
    return fake_saxs


def test_worker_writes_metric_failed(tmp_path, fake_pipeline):
    fake_pipeline.curve = flat_curve()
    cfg_path, outdir = _worker_cfg(tmp_path, {})
    assert worker.main(["worker.py", str(cfg_path)]) == 0
    assert not (outdir / "DONE").exists() and not (outdir / "FAILED").exists()
    data = json.loads((outdir / "METRIC_FAILED").read_text())
    assert "no simulated peak" in data["reason"]
    assert data["result"]["n_pairs_below_rmin"] == "2"
    assert json.loads((outdir / "shift_rmse_diagnostics.json").read_text())["failed"] is True


def test_worker_done_with_no_peak_fallback(tmp_path, fake_pipeline):
    fake_pipeline.curve = flat_curve()
    cfg_path, outdir = _worker_cfg(tmp_path, {"no_peak": "fallback"})
    assert worker.main(["worker.py", str(cfg_path)]) == 0
    done = json.loads((outdir / "DONE").read_text())
    diag = json.loads((outdir / "shift_rmse_diagnostics.json").read_text())
    assert done["loss"] == pytest.approx(diag["loss"])
    assert diag["fallbacks"]["peak_sim_max_fallback"] is True


def test_worker_other_errors_still_failed(tmp_path, monkeypatch):
    def boom(outdir, **kw):
        raise RuntimeError("GPU init failed")
    monkeypatch.setattr(worker, "run_simulation", boom)
    cfg_path, outdir = _worker_cfg(tmp_path, {})
    assert worker.main(["worker.py", str(cfg_path)]) == 1
    assert (outdir / "FAILED").exists() and not (outdir / "METRIC_FAILED").exists()


def _job(tmp_path, ds_id, status):
    sim_dir = tmp_path / ds_id
    sim_dir.mkdir(parents=True, exist_ok=True)
    return sp.Job(idx=0, name=f"j_{ds_id}", ds_id=ds_id, sim_dir=sim_dir,
                  config_path=sim_dir / "c.json", sbatch_path=sim_dir / "s.sbatch",
                  out_path=sim_dir / "o.out", done_status=status)


def test_inspect_flags_reads_metric_failed(tmp_path):
    job = _job(tmp_path, "d0", None)
    (job.sim_dir / "METRIC_FAILED").write_text(json.dumps(
        {"run_time_seconds": 12.5, "host": "g3045", "reason": "x"}))
    sp.inspect_flags(job)
    assert job.done_status == "METRIC_FAILED"
    assert job.run_time_seconds == 12.5 and job.host == "g3045"


def test_retry_skips_metric_failed(tmp_path, monkeypatch):
    calls = []

    def fake_submit(specs, cfg, **kw):
        specs = list(specs)
        calls.append([s["ds_id"] for s in specs])
        if len(calls) == 1:
            return [_job(tmp_path, "d0", "METRIC_FAILED"),
                    _job(tmp_path, "d1", "FAILED"),
                    _job(tmp_path, "d2", "DONE")]
        return [_job(tmp_path, s["ds_id"], "DONE") for s in specs]

    monkeypatch.setattr(sp, "submit_jobs", fake_submit)
    specs = [{"name": f"j{i}", "ds_id": f"d{i}", "worker_config": {}} for i in range(3)]
    cfg = sp.LauncherConfig(run_dir=tmp_path / "_jobs")
    jobs = sp.submit_jobs_with_retry(specs, cfg, max_job_retries=2)
    assert calls == [["d0", "d1", "d2"], ["d1"]]
    assert [j.done_status for j in jobs] == ["METRIC_FAILED", "DONE", "DONE"]


def test_collector_fails_eval_on_metric_failed(tmp_path):
    out_root = tmp_path / "run"
    out_root.mkdir()
    jobs = [_job(tmp_path, "d0", "DONE"), _job(tmp_path, "d1", "METRIC_FAILED")]
    (jobs[0].sim_dir / "DONE").write_text(json.dumps({"loss": 0.5, "result": {}}))
    (jobs[1].sim_dir / "METRIC_FAILED").write_text(json.dumps({"reason": "no peak"}))

    class _DS:
        def __init__(self, i):
            self.id, self.weight = i, 1.0

    plan = dict(density=0.005, n=12.0, m=6.0, r0=2.5, U0=3.0)
    plans = [dict(plan, ds=_DS("d0")), dict(plan, ds=_DS("d1"))]
    total, ok, records = bo._collect_parallel_eval_loss(
        eval_id=4, G={}, plans=plans, finished_jobs=jobs, out_root=str(out_root),
    )
    assert ok is False
    assert [r["loss"] for r in records] == [0.5, "METRIC_FAILED"]
    assert "eval 4 METRIC_FAILED: no peak" in (out_root / "error_d1.txt").read_text()

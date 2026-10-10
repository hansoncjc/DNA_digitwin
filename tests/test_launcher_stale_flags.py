"""Stale worker flags and the trim_tail default."""
import inspect

import numpy as np
import pytest

import bo
from datasets import Dataset
from parallel import submit_parallel as sp


def _spec(sim_dir, name="eval_007_d0"):
    return {"name": name, "ds_id": "d0", "worker_config": {"outdir": str(sim_dir)}}


@pytest.mark.parametrize("flag", ["DONE", "METRIC_FAILED", "FAILED", "RUNNING"])
def test_prepare_jobs_removes_stale_flag(tmp_path, flag):
    # A master killed mid-evaluation leaves the old candidate's flag in the
    # eval directory that the resumed run reuses.
    sim_dir = tmp_path / "eval_007" / "d0"
    sim_dir.mkdir(parents=True)
    (sim_dir / flag).write_text('{"loss": 0.123}')
    cfg = sp.LauncherConfig(run_dir=tmp_path / "_jobs")
    sp.make_run_dir(cfg.run_dir, clean=True)

    jobs = sp.prepare_jobs(cfg, [_spec(sim_dir)])

    assert not (sim_dir / flag).exists()
    sp.inspect_flags(jobs[0])
    assert jobs[0].done_status is None


def test_prepare_jobs_keeps_other_files(tmp_path):
    sim_dir = tmp_path / "eval_007" / "d0"
    sim_dir.mkdir(parents=True)
    (sim_dir / "DONE").write_text("{}")
    (sim_dir / "sim_params_d0.csv").write_text("x\n1\n")
    cfg = sp.LauncherConfig(run_dir=tmp_path / "_jobs")
    sp.make_run_dir(cfg.run_dir, clean=True)

    sp.prepare_jobs(cfg, [_spec(sim_dir)])

    assert (sim_dir / "sim_params_d0.csv").exists()


def test_trim_tail_defaults_to_zero(tmp_path):
    assert inspect.signature(bo.make_global_objective).parameters["trim_tail"].default == 0
    assert inspect.signature(Dataset.load_exp_curve).parameters["trim_tail"].default == 0

    curve = np.column_stack([np.linspace(1e-3, 0.13, 508), np.ones(508)])
    path = tmp_path / "target_sq.npy"
    np.save(path, curve)
    ds = Dataset(id="d0", exp_path=path)
    assert ds.load_exp_curve().shape[0] == 508
    assert ds.load_exp_curve(trim_tail=200).shape[0] == 308

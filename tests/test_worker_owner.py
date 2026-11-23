"""An older job cannot write into a folder a newer job took over, and every
evaluation computes its S(q) from the GSD its own simulation wrote
(TIMELINE 1.1, C32)."""
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import bo
import scattering
from parallel import submit_parallel as sp
from parallel import worker
from shift_rmse_curves import crystal_curve

OLD = "i7_d00:old"
NEW = "i7_d00_r1:new"
STALE_GSD = "DNA_assembly_20000101000000.gsd"      # sorts before this run's GSD


def _cfg(tmp_path, owner, outdir=None):
    exp_path = tmp_path / "target.npy"
    np.save(exp_path, crystal_curve())
    outdir = outdir or tmp_path / "eval_007" / "d0"
    cfg = {
        "outdir": str(outdir),
        "run_kwargs": {"density": 0.005, "U_0": 3.0, "r0": 2.5, "n": 12.0, "m": 6.0},
        "scattering": {"method": "saxsfft", "kwargs": {}},
        "loss": {
            "exp_path": str(exp_path), "trim_tail": 0, "datatype": "sq", "ffpath": "",
            "metric": "shift_rmse", "scattering_method": "saxsfft",
            "compare_q_range": None, "metric_kwargs": {"overlap_trim": 3},
        },
    }
    if owner is not None:
        cfg["owner"] = owner
    path = tmp_path / f"cfg_{len(list(tmp_path.glob('cfg_*.json')))}.json"
    path.write_text(json.dumps(cfg))
    return path, Path(outdir)


@pytest.fixture
def pipeline(monkeypatch):
    """Fake simulation (writes its own timestamped GSD) and fake S(q) that
    records the GSD it was given. ``state["during_sim"]`` / ``["during_sq"]``
    run inside them."""
    state = {"sim_calls": 0, "sq_paths": [], "during_sim": None, "during_sq": None}

    def fake_sim(outdir, **kw):
        state["sim_calls"] += 1
        gsd = Path(outdir) / "DNA_assembly_20261122120000.gsd"
        gsd.write_text("this run")
        if state["during_sim"]:
            state["during_sim"](Path(outdir))
        return {"gsd_path": str(gsd), "rmax": 4.9, "t_tol": 0.03, "n_pairs_below_rmin": 0}

    def fake_saxs(save_dir, path=None, **kw):
        state["sq_paths"].append(path)
        if state["during_sq"]:
            state["during_sq"](Path(save_dir))
        d = Path(save_dir) / "S(q)_data"
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "average_structure_factor.npy", crystal_curve(q1=0.0131, amp=6.0))

    monkeypatch.setattr(worker, "run_simulation", fake_sim)
    monkeypatch.setattr(worker, "convert_to_SAXS_fft", fake_saxs)
    return state


def _flags(outdir):
    return sorted(n for n in ("RUNNING", "DONE", "METRIC_FAILED", "FAILED") if (outdir / n).exists())


def test_owner_runs_as_before_and_reads_its_own_gsd(tmp_path, pipeline):
    cfg_path, outdir = _cfg(tmp_path, NEW)
    outdir.mkdir(parents=True)
    (outdir / "OWNER").write_text(NEW + "\n")
    (outdir / STALE_GSD).write_text("an older attempt")
    assert worker.main(["worker.py", str(cfg_path)]) == 0
    assert pipeline["sq_paths"] == [str(outdir / "DNA_assembly_20261122120000.gsd")]
    assert _flags(outdir) == ["DONE"]
    done = json.loads((outdir / "DONE").read_text())
    assert set(done) == {"run_time_seconds", "host", "loss", "result"}   # unchanged keys


def test_job_that_lost_the_folder_before_starting_writes_nothing(tmp_path, pipeline):
    cfg_path, outdir = _cfg(tmp_path, OLD)
    outdir.mkdir(parents=True)
    (outdir / "OWNER").write_text(NEW + "\n")
    (outdir / "DONE").write_text('{"loss": 0.42}')            # the newer job's result
    assert worker.main(["worker.py", str(cfg_path)]) == worker.NOT_OWNER_EXIT
    assert pipeline["sim_calls"] == 0
    assert (outdir / "DONE").read_text() == '{"loss": 0.42}'
    assert _flags(outdir) == ["DONE"] and not (outdir / "S(q)_data").exists()


def test_job_that_loses_the_folder_during_the_simulation_writes_no_sq_and_no_flag(tmp_path, pipeline):
    cfg_path, outdir = _cfg(tmp_path, OLD)
    outdir.mkdir(parents=True)
    (outdir / "OWNER").write_text(OLD + "\n")

    def taken_over(d):
        (d / "OWNER").write_text(NEW + "\n")
        (d / "RUNNING").write_text("newer job")
    pipeline["during_sim"] = taken_over
    assert worker.main(["worker.py", str(cfg_path)]) == worker.NOT_OWNER_EXIT
    assert pipeline["sq_paths"] == [] and not (outdir / "S(q)_data").exists()
    assert _flags(outdir) == ["RUNNING"]
    assert (outdir / "RUNNING").read_text() == "newer job"    # not removed


@pytest.mark.parametrize("fail", [False, True])
def test_job_that_loses_the_folder_before_its_flag_writes_no_flag(tmp_path, pipeline, monkeypatch, fail):
    cfg_path, outdir = _cfg(tmp_path, OLD)
    outdir.mkdir(parents=True)
    (outdir / "OWNER").write_text(OLD + "\n")

    def taken_over(d):
        (d / "OWNER").write_text(NEW + "\n")
        if fail:
            raise RuntimeError("S(q) failed")
    pipeline["during_sq"] = taken_over
    assert worker.main(["worker.py", str(cfg_path)]) == worker.NOT_OWNER_EXIT
    assert not any((outdir / n).exists() for n in ("DONE", "FAILED", "METRIC_FAILED"))


def test_config_without_owner_is_refused(tmp_path, pipeline):
    cfg_path, outdir = _cfg(tmp_path, None)
    outdir.mkdir(parents=True)
    assert worker.main(["worker.py", str(cfg_path)]) == 2
    assert pipeline["sim_calls"] == 0 and list(outdir.iterdir()) == []


def test_scattering_path_kwarg_is_refused(tmp_path, pipeline):
    cfg_path, outdir = _cfg(tmp_path, NEW)
    cfg = json.loads(cfg_path.read_text())
    cfg["scattering"]["kwargs"] = {"path": "/elsewhere/x.gsd"}
    cfg_path.write_text(json.dumps(cfg))
    outdir.mkdir(parents=True)
    (outdir / "OWNER").write_text(NEW + "\n")
    assert worker.main(["worker.py", str(cfg_path)]) == 1
    assert "must not set 'path'" in (outdir / "FAILED").read_text()
    assert pipeline["sim_calls"] == 0


def test_given_up_attempt_cannot_write_into_its_retry(tmp_path, pipeline):
    """prepare_jobs for the retry takes the folder over; the first attempt's
    worker, started late, writes nothing; the retry's worker writes DONE."""
    sim_dir = tmp_path / "eval_007" / "d0"
    spec = {"name": "i7_d00", "ds_id": "d0", "worker_config": json.loads(_cfg(tmp_path, None, sim_dir)[0].read_text())}
    cfg1 = sp.LauncherConfig(run_dir=tmp_path / "eval_007" / "_jobs")
    sp.make_run_dir(cfg1.run_dir, clean=True)
    (first,) = sp.prepare_jobs(cfg1, [spec])
    cfg2 = sp.LauncherConfig(run_dir=tmp_path / "eval_007" / "_jobs" / "retry_1")
    sp.make_run_dir(cfg2.run_dir, clean=True)
    (retry,) = sp.prepare_jobs(cfg2, [dict(spec, name="i7_d00_r1")])

    assert first.owner.startswith("i7_d00:") and retry.owner.startswith("i7_d00_r1:")
    assert (sim_dir / "OWNER").read_text().strip() == retry.owner
    assert json.loads(retry.config_path.read_text())["owner"] == retry.owner

    assert worker.main(["worker.py", str(first.config_path)]) == worker.NOT_OWNER_EXIT
    assert _flags(sim_dir) == [] and pipeline["sim_calls"] == 0
    assert worker.main(["worker.py", str(retry.config_path)]) == 0
    assert _flags(sim_dir) == ["DONE"]


def test_each_prepare_writes_a_new_token(tmp_path):
    sim_dir = tmp_path / "eval_007" / "d0"
    spec = {"name": "i7_d00", "ds_id": "d0", "worker_config": {"outdir": str(sim_dir)}}
    cfg = sp.LauncherConfig(run_dir=tmp_path / "_jobs")
    sp.make_run_dir(cfg.run_dir, clean=True)
    tokens = {sp.prepare_jobs(cfg, [spec])[0].owner for _ in range(3)}
    assert len(tokens) == 3                      # same name, new token each time


# --------------------------------------------- sequential path and scattering


class _DS:
    def __init__(self):
        self.id, self.weight, self.datatype = "d0", 1.0, "sq"
        self.exp_path = "/dev/null/d0.npy"
        self.sim = types.SimpleNamespace(n=12.0, m=6.0, density=0.005, r0=None, U0=None)

    def load_exp_curve(self, trim_tail=0):
        return crystal_curve(q1=0.0129)


def test_sequential_path_reads_its_own_gsd(tmp_path, monkeypatch):
    seen = []

    def fake_sim(density, U_0, r0, n, m, outdir, **kw):
        (Path(outdir) / STALE_GSD).write_text("older")
        gsd = Path(outdir) / "DNA_assembly_20261122120000.gsd"
        gsd.write_text("this run")
        return {"gsd_path": str(gsd), "rmax": 4.9, "t_tol": 0.03, "n_pairs_below_rmin": 0}

    def fake_saxs(save_dir, path=None, **kw):
        seen.append(path)
        d = Path(save_dir) / "S(q)_data"
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "average_structure_factor.npy", crystal_curve(q1=0.0131, amp=6.0))

    monkeypatch.setattr(bo, "run_simulation", fake_sim)
    monkeypatch.setattr(bo, "convert_to_SAXS_fft", fake_saxs)
    ps = bo.ParamSpace({"U0": {"bounds": (0.5, 8.0), "init": 3.0},
                        "r0": {"bounds": (2.0, 3.5), "init": 2.5},
                        "n": {"fixed": 12.0}, "m": {"fixed": 6.0}, "density": {"fixed": 0.005}})
    obj = bo.make_global_objective([_DS()], ps, ffpath="", out_root=str(tmp_path), mode="sim",
                                   metric="shift_rmse", compare_q_range=None)
    obj(ps.init_unit(), ffpath="")
    assert seen == [str(tmp_path / "eval_000" / "d0" / "DNA_assembly_20261122120000.gsd")]


def test_objective_refuses_a_scattering_path(tmp_path):
    ps = bo.ParamSpace({"U0": {"bounds": (0.5, 8.0), "init": 3.0}, "r0": {"fixed": 2.5},
                        "n": {"fixed": 12.0}, "m": {"fixed": 6.0}, "density": {"fixed": 0.005}})
    with pytest.raises(ValueError, match="must not set 'path'"):
        bo.make_global_objective([_DS()], ps, ffpath="", out_root=str(tmp_path), mode="sim",
                                 scattering_kwargs={"path": "x.gsd"})


def test_explicit_path_is_the_file_the_old_lookup_chose_in_a_fresh_folder(tmp_path, monkeypatch):
    """Default case: in a folder as run_simulation leaves it (one production
    GSD next to _init_lattice.gsd), convert_to_SAXS_fft(path=None) picks
    exactly the gsd_path that is now passed, so the S(q) is unchanged."""
    (tmp_path / "_init_lattice.gsd").write_text("init")
    gsd = tmp_path / "DNA_assembly_20261122120000.gsd"
    gsd.write_text("production")
    picked = []

    class _Picked(Exception):
        pass

    class FakeSF:
        def __init__(self, gsd_path, **kw):
            picked.append(gsd_path)
            raise _Picked

    monkeypatch.setitem(sys.modules, "saxsfft", types.SimpleNamespace(StructureFactor=FakeSF))
    for path in (None, str(gsd)):
        with pytest.raises(_Picked):
            scattering.convert_to_SAXS_fft(str(tmp_path), path=path)
    assert picked == [str(gsd), str(gsd)]

"""inverse.py: configuration, startup checks, resume and summary (HOOMD and botorch stubbed)."""
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

import bo
import inverse
from shift_rmse_curves import crystal_curve, flat_curve, gas_curve, saxsfft_q
from simulation import resolve_table_bounds
from test_resume import _fake_run_bo, _Killed


def _write_config(tmp_path, target, **over):
    cfg = {
        "description": "test",
        "target": str(target),
        "N": 5017,
        "density": 0.005,
        "search": {
            "U0": {"bounds": (0.5, 8.0), "init": 3.0},
            "r0": {"bounds": (2.0, 3.5), "init": 2.5},
            "n": {"bounds": (5.5, 15.0), "init": 12.0},
            "m_rel": {"bounds": (0.0, 1.0), "init": 2.0 / 7.0},
        },
        "max_evals": 3,
        "out_root": str(tmp_path / "out"),
        "mc_dfm_root": "",
        "parallel": False,
    }
    cfg.update(over)
    cfg = {k: v for k, v in cfg.items() if v is not ...}
    path = tmp_path / "config.py"
    path.write_text(f"CONFIG = {cfg!r}\n")
    return path


@pytest.fixture
def target(tmp_path):
    path = tmp_path / "target_sq.npy"
    np.save(path, crystal_curve(q1=0.0129))
    return path


@pytest.fixture
def pipeline(monkeypatch):
    state = {"calls": 0, "kill_at": None}

    def fake_sim(density, U_0, r0, n, m, outdir, **kw):
        i = state["calls"]
        state["calls"] += 1
        state["last_kw"] = kw
        if i == state["kill_at"]:
            raise _Killed()
        b = resolve_table_bounds(U_0, n, m, r0, N=kw["N"], density=density)
        return {"rmax": b["rmax"], "t_tol": b["t_tol"], "n_pairs_below_rmin": 0}

    def fake_saxs(save_dir, **kw):
        d = Path(save_dir) / "S(q)_data"
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "average_structure_factor.npy", crystal_curve(q1=0.0131, amp=6.0))

    monkeypatch.setattr(bo, "run_simulation", fake_sim)
    monkeypatch.setattr(bo, "convert_to_SAXS_fft", fake_saxs)
    log = []
    monkeypatch.setattr(bo, "run_bo", _fake_run_bo(log))
    state["run_bo_log"] = log
    return state


# ---------------- configuration ---------------- #

@pytest.mark.parametrize("over, match", [
    ({"extra": 1}, "unknown CONFIG keys"),
    ({"target_md5": "0" * 32}, "unknown CONFIG keys"),
    ({"N": ...}, "missing"),
    ({"metric": "mse"}, "available"),
    ({"potential": "lj"}, "available"),
    ({"scattering": "mcdfm"}, "available"),
    ({"compare_q_range": (0.003, 0.06)}, "must be None"),
    ({"batch_size": 10}, "batch_size > 1 needs parallel=True"),
    ({"batch_size": 0}, "batch_size must be >= 1"),
    ({"parallel": 1}, "True or False"),
    ({"parallel": True}, "needs parallel_cfg"),
    ({"parallel": True, "parallel_cfg": {"partition": "x"}}, "missing"),
    ({"parallel": True, "parallel_cfg": {"code_root": "x"}}, "unknown parallel_cfg"),
    ({"max_evals": 0}, "max_evals"),
    ({"metric_kwargs": {"lambda": 0.0}}, "metric_kwargs: Unknown shift_rmse metric_kwargs"),
    ({"metric_kwargs": {"no_peak": "skip"}}, "metric_kwargs: shift_rmse no_peak must be"),
    ({"metric_kwargs": [("lambda_shift", 0.0)]}, "metric_kwargs must be a dict"),
    ({"search": {"U0": {"fixed": 3.0}, "r0": {"fixed": 2.5}, "n": {"fixed": 12.0}}},
     "exactly one of m, m_rel"),
    ({"search": {"U0": {"fixed": 3.0}, "r0": {"fixed": 2.5}, "n": {"fixed": 12.0},
                 "m": {"fixed": 6.0}, "m_rel": {"fixed": 0.2}}}, "exactly one of m, m_rel"),
    ({"search": {"r0": {"fixed": 2.5}, "n": {"fixed": 12.0}, "m": {"fixed": 6.0}}}, "U0"),
    ({"search": {"U0": {"fixed": 3.0}, "r0": {"fixed": 2.5}, "n": {"fixed": 12.0},
                 "m": {"fixed": 6.0}, "density": {"fixed": 0.005}}}, "density"),
])
def test_config_errors(tmp_path, target, over, match):
    path = _write_config(tmp_path, target, **over)
    with pytest.raises(inverse.ConfigError, match=match):
        inverse.normalize_config(inverse.load_config(path))


def test_config_rewritten_in_the_same_second_is_read_again(tmp_path, target):
    # Same size and the same mtime: an import would reuse the cached
    # bytecode of the first version. load_config must read the file again.
    path = _write_config(tmp_path, target, max_evals=3)
    assert inverse.load_config(path)["max_evals"] == 3
    st = path.stat()
    path.write_text(path.read_text().replace("'max_evals': 3", "'max_evals': 2"))
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert path.stat().st_size == st.st_size
    assert inverse.load_config(path)["max_evals"] == 2
    assert not (tmp_path / "__pycache__").exists()


def test_example_config_is_valid():
    raw = inverse.load_config(
        inverse.REPO / "campaigns" / "inverse" / "inverse-example" / "config.py")
    cfg = inverse.normalize_config(raw)
    assert cfg["max_evals"] == 3 and cfg["parallel"] is False
    assert inverse.normalize_config(raw, {"parallel": True})["parallel"] is True


# ---------------- startup checks (nothing is simulated) ---------------- #

def _check_fails(tmp_path, pipeline, match, target_arr=None, target_path=None, **over):
    if target_arr is not None:
        target_path = tmp_path / "bad_target.npy"
        np.save(target_path, target_arr)
    path = _write_config(tmp_path, target_path, **over)
    with pytest.raises(inverse.ConfigError, match=match):
        inverse.run(path)
    assert pipeline["calls"] == 0
    assert not (tmp_path / "out").exists()


def test_missing_target(tmp_path, pipeline):
    _check_fails(tmp_path, pipeline, "target not found", target_path=tmp_path / "nope.npy")


def test_dispersed_target_continues(tmp_path, pipeline, capsys):
    target = tmp_path / "gas.npy"
    np.save(target, gas_curve())
    out = inverse.run(_write_config(tmp_path, target), check=True)
    assert out["info"]["target"]["dispersed"] is True
    assert out["info"]["target"]["q1"] is None
    assert "WARNING: target is dispersed" in capsys.readouterr().out
    summary = inverse.run(_write_config(tmp_path, target, max_evals=2))
    assert summary["n_successful"] == 2 and pipeline["calls"] == 2


def test_target_without_peak(tmp_path, pipeline):
    _check_fails(tmp_path, pipeline, "no first peak", target_arr=flat_curve(level=2.0))


def test_target_n_grid_mismatch(tmp_path, pipeline):
    q = saxsfft_q(n_grid=900)
    _check_fails(tmp_path, pipeline, "N_grid", target_arr=crystal_curve(q1=0.0129, q=q))


def test_target_short_of_the_asymptote_band(tmp_path, pipeline):
    # N_grid = 300 ends at q ~ 0.066, below the (0.085, 0.100) band.
    q = saxsfft_q(n_grid=300)
    _check_fails(tmp_path, pipeline, "asymptote band", target_arr=crystal_curve(q1=0.0129, q=q))


def test_resume_refuses_a_changed_target(tmp_path, pipeline, target):
    path = _write_config(tmp_path, target, max_evals=1)
    inverse.run(path)
    np.save(target, crystal_curve(q1=0.0135))     # same path, other content
    calls = pipeline["calls"]
    with pytest.raises(inverse.ConfigError, match="target_md5"):
        inverse.run(path)
    assert pipeline["calls"] == calls


def test_rmax_violation_on_the_box(tmp_path, pipeline, target):
    # N = 500 at density 0.005 gives L/2 = 23.2; r0 up to 10 pushes rmax past it.
    search = {"U0": {"bounds": (0.5, 8.0)}, "r0": {"bounds": (2.0, 10.0)},
              "n": {"bounds": (5.5, 15.0)}, "m_rel": {"bounds": (0.0, 1.0)}}
    _check_fails(tmp_path, pipeline, "minimum-image", target_path=target, N=500, search=search)


def test_box_with_n_not_above_m(tmp_path, pipeline, target):
    search = {"U0": {"fixed": 3.0}, "r0": {"fixed": 2.5},
              "n": {"bounds": (6.0, 12.0)}, "m": {"bounds": (4.0, 8.0)}}
    _check_fails(tmp_path, pipeline, "n <= m", target_path=target, search=search)


def test_m_rel_rules_reach_the_check(tmp_path, pipeline, target):
    search = {"U0": {"fixed": 3.0}, "r0": {"fixed": 2.5},
              "n": {"bounds": (5.0, 12.0)}, "m_rel": {"bounds": (0.0, 1.0)}}
    _check_fails(tmp_path, pipeline, "n > 5", target_path=target, search=search)


def test_main_reports_failure(tmp_path, pipeline, capsys):
    path = _write_config(tmp_path, tmp_path / "nope.npy")
    assert inverse.main([str(path)]) == 2
    assert "nothing was simulated" in capsys.readouterr().err


def test_check_only_writes_nothing(tmp_path, pipeline, target):
    path = _write_config(tmp_path, target)
    out = inverse.run(path, check=True)
    assert out["info"]["target"]["md5"] == inverse.file_md5(target)
    assert out["info"]["rmax"]["rmax"] < out["info"]["rmax"]["L_half"]
    assert out["resume"]["next_eval_id"] == 0
    assert pipeline["calls"] == 0 and not (tmp_path / "out").exists()


# ---------------- run, resume, summary ---------------- #

def test_run_resume_and_summary(tmp_path, pipeline, target):
    path = _write_config(tmp_path, target)
    pipeline["kill_at"] = 1                       # killed inside eval_001
    with pytest.raises(_Killed):
        inverse.run(path)
    out = tmp_path / "out"
    snap = json.loads((out / "inverse_config.json").read_text())
    assert snap["target"]["md5"] == inverse.file_md5(target)
    assert snap["locked"]["search"]["m_rel"]["bounds"] == [0.0, 1.0]
    assert len(snap["starts"]) == 1
    assert pipeline["last_kw"]["N"] == 5017 and "steps" not in pipeline["last_kw"]

    # Changing the box is refused before anything runs.
    calls = pipeline["calls"]
    bad = _write_config(tmp_path, target, search={**inverse.load_config(path)["search"],
                                                  "r0": {"bounds": (2.0, 3.0), "init": 2.5}})
    with pytest.raises(inverse.ConfigError, match="r0"):
        inverse.run(bad)
    assert pipeline["calls"] == calls

    # max_evals may change; the resume skips the abandoned eval_001.
    pipeline["kill_at"] = None
    path = _write_config(tmp_path, target, max_evals=4)
    summary = inverse.run(path)
    resumed = pipeline["run_bo_log"][-1]
    assert resumed["warm_start"][0].shape[0] == 1 and resumed["max_evals"] == 4
    assert resumed["batch_size"] == 1
    assert summary["n_successful"] == 4 and summary["n_abandoned"] == 1
    assert summary["complete"] and summary["max_evals"] == 4
    best = summary["best"]
    assert best["loss"] == pytest.approx(best["rmse"] + best["shift"])
    assert best["m"] == pytest.approx(4.0 + best["m_rel"] * (best["n"] - 5.0))
    assert best["density"] == 0.005
    assert (out / "inverse_summary.txt").read_text().startswith("Inverse design summary")
    assert len(json.loads((out / "inverse_config.json").read_text())["starts"]) == 2

    # --summary rewrites it without running anything.
    calls = pipeline["calls"]
    again = inverse.run(path, summary_only=True)
    assert again["best"] == best and pipeline["calls"] == calls


def test_overrides_for_smoke_tests(tmp_path, pipeline, target):
    path = _write_config(tmp_path, target)
    inverse.run(path, {"out_root": str(tmp_path / "smoke"), "steps": 1000, "device": "cpu"})
    assert pipeline["last_kw"]["steps"] == 1000 and pipeline["last_kw"]["device"] == "cpu"
    snap = json.loads((tmp_path / "smoke" / "inverse_config.json").read_text())
    assert snap["locked"]["steps"] == 1000
    assert isinstance(snap["locked"]["steps"], int)
    # A resume without --steps would change the run: refused.
    with pytest.raises(inverse.ConfigError, match="steps: run started with 1000, now None"):
        inverse.run(path, {"out_root": str(tmp_path / "smoke")})


def _move_run(tmp_path, path, src, dst, old_format=False):
    inverse.run(path, {"out_root": str(src)})
    if old_format:   # inverse_config.json written before 2026-10-11
        snap = json.loads((src / "inverse_config.json").read_text())
        snap["locked"]["gp_log_dir"] = str(src / "gp_log")
        (src / "inverse_config.json").write_text(json.dumps(snap))
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


@pytest.mark.parametrize("old_format", [False, True])
def test_moved_run_folder_resumes(tmp_path, pipeline, target, old_format):
    path = _write_config(tmp_path, target, max_evals=1)
    src, dst = tmp_path / "a" / "run", tmp_path / "b" / "c" / "run"
    _move_run(tmp_path, path, src, dst, old_format)
    path = _write_config(tmp_path, target, max_evals=2)
    summary = inverse.run(path, {"out_root": str(dst)})
    assert summary["n_successful"] == 2 and pipeline["calls"] == 2
    snap = json.loads((dst / "inverse_config.json").read_text())
    assert [s["out_root"] for s in snap["starts"]] == [str(src.resolve()), str(dst.resolve())]
    if not old_format:
        assert snap["locked"]["gp_log_dir"] == "gp_log"


def test_changed_gp_log_dir_refused(tmp_path, pipeline, target):
    path = _write_config(tmp_path, target, max_evals=1)
    inverse.run(path)
    path = _write_config(tmp_path, target, max_evals=2,
                         gp_log_dir=str(tmp_path / "out" / "other_log"))
    with pytest.raises(inverse.ConfigError, match="gp_log_dir: run started with 'gp_log'"):
        inverse.run(path)
    assert pipeline["calls"] == 1


# ---------------- metric_kwargs (R6.8, 2026-10-14) ---------------- #

def test_metric_kwargs_default_is_the_old_behaviour(tmp_path, target):
    cfg = inverse.normalize_config(inverse.load_config(_write_config(tmp_path, target)))
    assert cfg["metric_kwargs"] == {}
    cfg["target_md5"] = inverse.file_md5(target)     # set by the startup checks
    assert inverse.locked_fields(cfg)["metric_kwargs"] == {}
    assert "lambda_shift = 1" in inverse._loss_text(cfg)


@pytest.mark.parametrize("lam", [0.0, 2.0])
def test_lambda_shift_reaches_the_loss(tmp_path, pipeline, target, lam):
    base = inverse.run(_write_config(tmp_path, target, max_evals=1,
                                     out_root=str(tmp_path / "default")))["best"]
    path = _write_config(tmp_path, target, max_evals=1, out_root=str(tmp_path / "lam"),
                         metric_kwargs={"lambda_shift": lam})
    summary = inverse.run(path)
    best = summary["best"]
    assert base["shift"] > 0
    assert best["rmse"] == pytest.approx(base["rmse"])         # alignment unchanged
    assert best["shift"] == pytest.approx(lam * base["shift"])
    assert best["loss"] == pytest.approx(best["rmse"] + best["shift"])
    assert summary["metric_kwargs"] == {"lambda_shift": lam}
    assert f"lambda_shift = {lam:g}" in summary["loss"]
    snap = json.loads((tmp_path / "lam" / "inverse_config.json").read_text())
    assert snap["locked"]["metric_kwargs"] == {"lambda_shift": lam}
    assert snap["target"]["metric_kwargs"] == {"lambda_shift": lam}


def test_metric_kwargs_locked_on_resume(tmp_path, pipeline, target):
    inverse.run(_write_config(tmp_path, target, max_evals=1,
                              metric_kwargs={"lambda_shift": 0.0}))
    path = _write_config(tmp_path, target, max_evals=2, metric_kwargs={"lambda_shift": 2.0})
    with pytest.raises(inverse.ConfigError, match="metric_kwargs: run started with"):
        inverse.run(path)
    assert pipeline["calls"] == 1
    path = _write_config(tmp_path, target, max_evals=2, metric_kwargs={"lambda_shift": 0.0})
    assert inverse.run(path)["n_successful"] == 2


def test_snapshot_without_metric_kwargs_resumes_with_defaults(tmp_path, pipeline, target):
    """inverse_config.json written before metric_kwargs existed (R6.4-R6.6)."""
    inverse.run(_write_config(tmp_path, target, max_evals=1))
    snap_path = tmp_path / "out" / "inverse_config.json"
    snap = json.loads(snap_path.read_text())
    del snap["locked"]["metric_kwargs"]
    snap_path.write_text(json.dumps(snap))
    path = _write_config(tmp_path, target, max_evals=2, metric_kwargs={"lambda_shift": 0.0})
    with pytest.raises(inverse.ConfigError, match="metric_kwargs: run started with {}"):
        inverse.run(path)
    assert inverse.run(_write_config(tmp_path, target, max_evals=2))["n_successful"] == 2

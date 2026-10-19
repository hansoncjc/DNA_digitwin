"""R6.9 r0 scan: grid, checks before submission, and one point with a fake simulation."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import scattering
import simulation

CAMPAIGNS = Path(__file__).resolve().parents[1] / "campaigns"
sys.path.insert(0, str(CAMPAIGNS / "r6.9-hs-r0-scan"))
import scan_config as r69  # noqa: E402
import run_point  # noqa: E402


def test_grid():
    pts = r69.POINTS
    assert [p["id"] for p in pts] == [f"s{i}" for i in range(13)]
    assert pts[0]["role"].startswith("control") and pts[0]["r0"] == pytest.approx(1.3445141132218734)
    scan = pts[1:]
    assert {(p["r0"], p["U0"]) for p in scan} == {(r, u) for r in (1.07, 1.11, 1.15, 1.20)
                                                   for u in (1.8, 2.1, 2.4)}
    assert all(p["n"] == 15.0 and p["m"] == 4.0 for p in pts)
    assert r69.METRIC_KWARGS == {}              # lambda 1, as R6.6


def test_target_is_the_r66_target():
    assert run_point.check_target() == "37c39120c6d44871953769456f8f3b75"
    sys.path.insert(0, str(CAMPAIGNS / "inverse" / "r6.6-hs-fluid"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "r66_config", CAMPAIGNS / "inverse" / "r6.6-hs-fluid" / "config.py")
    r66 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r66)
    assert Path(r66.CONFIG["target"]).resolve() == r69.TARGET.resolve()
    assert r66.CONFIG["N"] == r69.N and r66.CONFIG["density"] == r69.DENSITY


def test_list_reports_no_failure(capsys):
    assert run_point.list_points() == 0
    assert "13 points, 0 failed" in capsys.readouterr().out


def test_bad_point_is_refused():
    with pytest.raises(ValueError, match="must exceed"):
        run_point.bounds({"U0": 2.0, "r0": 1.1, "n": 4.0, "m": 4.0})


def test_run_one_with_fake_simulation(tmp_path, monkeypatch):
    target = np.load(r69.TARGET)
    seen = {}

    def fake_sim(density, U_0, r0, n, m, outdir, *, seed=42, **kw):   # seed as run_simulation
        seen.update(density=density, U_0=U_0, r0=r0, n=n, m=m, kw=kw)
        return {"rmax": 3.0, "n_pairs_below_rmin": 0}

    def fake_saxs(save_dir, **kw):
        d = Path(save_dir) / "S(q)_data"
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "average_structure_factor.npy", target)    # the target itself: loss 0

    monkeypatch.setattr(simulation, "run_simulation", fake_sim)
    monkeypatch.setattr(scattering, "convert_to_SAXS_fft", fake_saxs)
    run_point.run_one(3, tmp_path)
    p = r69.POINTS[3]
    d = tmp_path / p["id"] / "d0"
    assert seen["U_0"] == p["U0"] and seen["r0"] == p["r0"] and seen["n"] == 15.0 and seen["m"] == 4.0
    assert seen["density"] == 0.005 and seen["kw"] == {"N": 5000, "plot": False}
    diag = json.loads((d / "shift_rmse_diagnostics.json").read_text())
    assert diag["loss"] == pytest.approx(0.0, abs=1e-12) and diag["q1_ratio"] == pytest.approx(1.0)
    rec = json.loads((d / "point_params.json").read_text())
    assert rec["point"]["id"] == p["id"] and rec["seed"] == 42 and rec["metric_kwargs"] == {}
    assert rec["target_md5"] == r69.TARGET_MD5 and "sim_result" in rec

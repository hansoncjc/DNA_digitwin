"""R4 (training on measured curves): config, startup checks, and both objective
paths on synthetic I(q) files (stub simulation, stub botorch, stub launcher)."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import bo
from datasets import check_mapped_params, physics_map
from metrics import shift_rmse_params
from parallel import submit_parallel as sp
from shift_rmse_curves import crystal_curve
from test_batch_bo import fake_botorch  # noqa: F401

CAMPAIGNS = Path(__file__).resolve().parents[1] / "campaigns"
sys.path.insert(0, str(CAMPAIGNS / "r4-measured"))
import r4_config as r4  # noqa: E402
import run_r4  # noqa: E402

sys.path.insert(0, str(CAMPAIGNS / "r3.2-physics-validate"))
import train_config as r32  # noqa: E402


def test_config():
    assert [c["id"] for c in r4.CONDITIONS] == [f"d{i}" for i in range(6)]
    assert {(c["L_bridge"], c["C_anchor"]) for c in r4.CONDITIONS} == {
        (20.0, 20.0), (20.0, 60.0), (20.0, 100.0), (20.0, 140.0), (40.0, 100.0), (80.0, 100.0)}
    assert len({c["file"] for c in r4.CONDITIONS}) == 6
    assert r4.C_NACL == 10.0
    assert r4.PARAM_CFG is r32.PARAM_CFG and r4.N == r32.N and r4.DENSITY == r32.DENSITY
    p = shift_rmse_params(r4.METRIC_KWARGS)
    assert p["intensity_scale"] == "none" and p["dispersed_delta"] == 0.25
    assert (r4.DATATYPE, r4.TRIM_TAIL, r4.MAX_EVALS) == ("iq", 200, 100)
    assert (run_r4.REPO / r4.FORMFACTOR).exists()


@pytest.mark.parametrize("cond", r4.CONDITIONS, ids=lambda c: c["id"])
def test_initial_point_inside_limits(cond):
    init = {k: v["init"] for k, v in r4.PARAM_CFG.items()}
    p = physics_map(cond["L_bridge"], cond["C_anchor"], r4.C_NACL, **init)
    check_mapped_params(p["U0"], p["n"], p["m"])


def test_c_anchor_20_is_only_partly_inside_the_box():
    fractions = {c["id"]: run_r4.box_fraction(c, n=512) for c in r4.CONDITIONS}
    assert 0.3 < fractions["d0"] < 0.7
    assert all(fractions[i] == 1.0 for i in ("d1", "d2", "d3", "d4", "d5"))


# ------------------------------------------------- synthetic measured curves

def _write_iq(folder, peak_amp=1.5, n=490):
    """I(q) = S(q) P(q) on a measured-like q grid (0.0028-0.145, n points, 3 columns)."""
    ff = np.loadtxt(run_r4.REPO / r4.FORMFACTOR, skiprows=1)
    q = np.linspace(0.0028, 0.145, n)
    s = 1.0 + peak_amp * np.exp(-0.5 * (np.log(q / 0.0127) / 0.12) ** 2) - 0.4 * np.exp(
        -0.5 * (np.log(q / 0.0075) / 0.25) ** 2)
    p = np.exp(np.interp(np.log(q), np.log(ff[:, 0]), np.log(ff[:, 1])))
    folder.mkdir(parents=True, exist_ok=True)
    md5 = {}
    for c in r4.CONDITIONS:
        arr = np.column_stack([q, s * p, 0.03 * s * p])
        np.save(folder / c["file"], arr)
        md5[c["id"]] = hashlib.md5((folder / c["file"]).read_bytes()).hexdigest()
    return md5


@pytest.fixture
def data(tmp_path, monkeypatch):
    folder = tmp_path / "I(q)"
    md5 = _write_iq(folder)
    monkeypatch.setattr(r4, "CONDITIONS", [dict(c, md5=md5[c["id"]]) for c in r4.CONDITIONS])
    return folder


def test_startup_checks_pass(data, tmp_path):
    out = tmp_path / "out"
    run_r4.main(["--data", str(data), "--out", str(out), "--check-only", "--mc-dfm-root", ""])
    rep = json.loads((out / "r4_startup_checks.json").read_text())
    assert rep["problems"] == [] and len(rep["conditions"]) == 6
    for row in rep["conditions"]:
        assert row["q1"] == pytest.approx(0.0127, rel=0.03)
        assert row["dispersed"] is False and row["n_grid_estimate"] is None
        assert row["q_max"] < 0.09    # trim_tail = 200


def test_startup_checks_fail_on_md5_and_dispersed(tmp_path, monkeypatch):
    folder = tmp_path / "I(q)"
    _write_iq(folder, peak_amp=0.05)          # flat: dispersed at delta 0.25
    with pytest.raises(SystemExit) as exc:
        run_r4.main(["--data", str(folder), "--out", str(tmp_path / "o"), "--check-only",
                     "--mc-dfm-root", ""])
    msg = str(exc.value)
    assert "md5" in msg and "dispersed" in msg


# ------------------------------------------------- objective, both paths

@pytest.fixture
def stub_forward(monkeypatch):
    def fake_sim(density, U_0, r0, n, m, outdir, **kw):
        return {"rmax": 3.0 * r0, "t_tol": 0.03, "n_pairs_below_rmin": 0}

    def fake_saxs(save_dir, **kw):
        d = Path(save_dir) / "S(q)_data"
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "average_structure_factor.npy", crystal_curve(q1=0.0131, amp=6.0))

    monkeypatch.setattr(bo, "run_simulation", fake_sim)
    monkeypatch.setattr(bo, "convert_to_SAXS_fft", fake_saxs)


def test_sequential_smoke(data, tmp_path, stub_forward, fake_botorch):  # noqa: F811
    out = tmp_path / "out"
    run_r4.main(["--data", str(data), "--out", str(out), "--smoke", "--sequential",
                 "--mc-dfm-root", ""])
    blocks = [l for l in (out / "bo_trajectory.csv").read_text().splitlines()
              if l.startswith("# Iteration")]
    ok = [int(l.split()[2].split(",")[0]) for l in blocks if ",total_loss," in l]
    assert len(ok) == 2
    # a failed block can only be a point that maps C_anchor = 20 below U0 = 0.5
    if len(blocks) > 2:
        assert "Mapped parameters outside the checked range: U0=" in (out / "error_d0.txt").read_text()
    for eval_id in ok:
        for ds in ("d0", "d5"):
            diag = json.loads((out / f"eval_{eval_id:03d}" / ds / "shift_rmse_diagnostics.json")
                              .read_text())
            assert diag["intensity_scale"] == "none" and diag["anchor_scale"] == 1.0
            assert diag["params"]["dispersed_delta"] == 0.25
            assert diag["aligned"] is True and diag["window_abs"][1] < 0.09


def test_parallel_job_specs_carry_the_r4_settings(data, tmp_path, stub_forward,
                                                  fake_botorch, monkeypatch):  # noqa: F811
    import parallel
    specs = []

    def submit(job_specs, cfg, **kw):
        jobs = []
        for s in job_specs:
            specs.append(s)
            sim_dir = Path(s["worker_config"]["outdir"])
            (sim_dir / "DONE").write_text(json.dumps({
                "loss": 0.5, "result": {"rmax": 5.0, "t_tol": 0.03, "n_pairs_below_rmin": 0}}))
            jobs.append(sp.Job(idx=0, name=s["name"], ds_id=s["ds_id"], sim_dir=sim_dir,
                               config_path=sim_dir / "c.json", sbatch_path=sim_dir / "s.sbatch",
                               out_path=sim_dir / "o.out", done_status="DONE"))
        return jobs

    monkeypatch.setattr(parallel, "submit_jobs_with_retry", submit)
    out = tmp_path / "out"
    run_r4.main(["--data", str(data), "--out", str(out), "--smoke", "--mc-dfm-root", ""])
    # two successful evaluations, each submitted as both conditions at once; a point
    # that maps C_anchor = 20 below U0 = 0.5 fails before submission (no GPU job)
    names = sorted(s["name"] for s in specs)
    evals = sorted({n[:3] for n in names})
    assert len(evals) == 2 and names == sorted(f"{e}_d{d}" for e in evals for d in ("00", "05"))
    for s in specs:
        loss = s["worker_config"]["loss"]
        assert (loss["datatype"], loss["trim_tail"]) == ("iq", 200)
        assert (loss["q_min"], loss["q_max"]) == r4.EXTRACT_Q_WINDOW
        assert loss["metric"] == "shift_rmse" and loss["compare_q_range"] is None
        assert loss["metric_kwargs"] == r4.METRIC_KWARGS
        assert Path(loss["ffpath"]) == run_r4.REPO / r4.FORMFACTOR
        assert s["worker_config"]["run_kwargs"]["N"] == 5000

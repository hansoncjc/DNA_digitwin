"""Failed trajectory rows record the simulated parameters in both paths (C13)."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import bo
from parallel import submit_parallel as sp
from shift_rmse_curves import crystal_curve
from simulation import resolve_table_bounds


class _DS:
    def __init__(self, ds_id="d0", U0=3.0):
        self.id = ds_id
        self.weight = 1.0
        self.datatype = "sq"
        self.exp_path = f"/dev/null/{ds_id}.npy"
        self.sim = SimpleNamespace(n=12.0, m=6.0, density=0.005, r0=2.5, U0=U0)

    def load_exp_curve(self, trim_tail=0):
        return crystal_curve(q1=0.0129)


@pytest.fixture
def pipeline(monkeypatch):
    state = {"fail": set()}

    def fake_sim(density, U_0, r0, n, m, outdir, **kw):
        if Path(outdir).name in state["fail"]:
            raise RuntimeError("simulated HOOMD failure")
        b = resolve_table_bounds(U_0, n, m, r0, N=5000, density=density)
        return {"rmax": b["rmax"], "t_tol": b["t_tol"], "n_pairs_below_rmin": 0}

    def fake_saxs(save_dir, **kw):
        d = Path(save_dir) / "S(q)_data"
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "average_structure_factor.npy", crystal_curve(q1=0.0131, amp=6.0))

    monkeypatch.setattr(bo, "run_simulation", fake_sim)
    monkeypatch.setattr(bo, "convert_to_SAXS_fft", fake_saxs)
    return state


def _failed_rows(path):
    """Rows of FAILED blocks, read with _TRAJECTORY_PARAM_COLUMNS (they carry no header)."""
    rows, in_failed = [], False
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if row and row[0].startswith("# Iteration"):
                in_failed = row[1] == "EVALUATION"
            elif in_failed and row and row[0] != "iteration":
                rows.append(dict(zip(bo._TRAJECTORY_PARAM_COLUMNS, row)))
    return rows


def _objective(out, datasets):
    ps = bo.ParamSpace({"n": {"bounds": (11.0, 13.0), "init": 12.0},
                        "m_rel": {"bounds": (0.0, 1.0), "init": 2.0 / 7.0}})
    obj = bo.make_global_objective(
        datasets, ps, ffpath="", out_root=str(out), trim_tail=0, mode="sim",
        metric="shift_rmse", compare_q_range=None,
    )
    return obj, ps


def test_sequential_failure_records_simulated_values(tmp_path, pipeline):
    pipeline["fail"] = {"d1"}
    obj, ps = _objective(tmp_path, [_DS("d0", 1.0), _DS("d1", 6.0)])
    with pytest.raises(bo.EvaluationFailed):
        obj(ps.init_unit(), ffpath="")

    rows = {r["dataset_id"]: r for r in _failed_rows(tmp_path / "bo_trajectory.csv")}
    assert set(rows) == {"d0", "d1"}
    assert rows["d1"]["loss"] == "FAILED"
    for ds_id, U0 in (("d0", 1.0), ("d1", 6.0)):
        r = rows[ds_id]
        assert (float(r["density"]), float(r["r0"]), float(r["U0"])) == (0.005, 2.5, U0)
        assert float(r["n"]) == pytest.approx(12.0)
        assert float(r["m"]) == pytest.approx(6.0)
        assert float(r["m_rel"]) == pytest.approx(2.0 / 7.0)


def test_resolution_failure_still_writes_error(tmp_path, pipeline):
    ds = _DS("d0")
    ds.sim.r0 = None                      # r0 neither in the ParamSpace nor in dataset.sim
    obj, ps = _objective(tmp_path, [ds])
    with pytest.raises(bo.EvaluationFailed):
        obj(ps.init_unit(), ffpath="")
    (row,) = _failed_rows(tmp_path / "bo_trajectory.csv")
    assert all(row[c] == "ERROR" for c in ("density", "n", "m", "r0", "U0"))
    assert float(row["m_rel"]) == pytest.approx(2.0 / 7.0)


def test_parallel_and_sequential_failed_rows_match(tmp_path, pipeline):
    pipeline["fail"] = {"d0"}
    obj, ps = _objective(tmp_path / "seq", [_DS("d0")])
    with pytest.raises(bo.EvaluationFailed):
        obj(ps.init_unit(), ffpath="")
    (seq,) = _failed_rows(tmp_path / "seq" / "bo_trajectory.csv")

    G = ps.decode(ps.unit_to_phys(ps.init_unit()))
    sim_dir = tmp_path / "par" / "eval_000" / "d0"
    sim_dir.mkdir(parents=True)
    job = sp.Job(idx=0, name="j", ds_id="d0", sim_dir=sim_dir, config_path=sim_dir / "c",
                 sbatch_path=sim_dir / "s", out_path=sim_dir / "o", done_status="FAILED")
    plan = dict(ds=_DS("d0"), density=0.005, n=float(seq["n"]), m=float(seq["m"]),
                r0=2.5, U0=3.0, save_dir=str(sim_dir))
    _, ok, (par,) = bo._collect_parallel_eval_loss(
        eval_id=0, G=G, plans=[plan], finished_jobs=[job], out_root=str(tmp_path / "par"))
    assert not ok
    for c in ("loss", "density", "n", "m", "r0", "U0", "m_rel"):
        assert str(par[c]) == seq[c]


def test_warm_start_skips_failed_rows_with_values(tmp_path, pipeline):
    obj, ps = _objective(tmp_path, [_DS("d0")])
    pipeline["fail"] = {"d0"}
    with pytest.raises(bo.EvaluationFailed):
        obj(ps.init_unit(), ffpath="")            # eval 0: failed, real values
    pipeline["fail"] = set()
    loss = float(obj(ps.init_unit(), ffpath=""))  # eval 1: success
    train_x, train_y, next_id = bo.load_warm_start_from_trajectory(
        tmp_path / "bo_trajectory.csv", ps)
    assert train_x.shape[0] == 1 and next_id == 2
    assert train_y.item() == pytest.approx(-loss)
    assert bo.inspect_resume(tmp_path)["failed_ids"] == [0]

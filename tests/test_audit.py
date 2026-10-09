"""Per-(eval, dataset) audit columns in bo_trajectory.csv and loss_components.txt."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import bo
from parallel import submit_parallel as sp
from shift_rmse_curves import crystal_curve
from simulation import resolve_table_bounds

W_NARROW = (0.004, 0.040)


class _DS:
    def __init__(self, ds_id, U0, q1):
        self.id = ds_id
        self.weight = 1.0
        self.datatype = "sq"
        self.exp_path = f"/dev/null/{ds_id}.npy"
        self.sim = SimpleNamespace(n=12.0, m=6.0, density=0.005, r0=2.5, U0=U0)
        self._curve = crystal_curve(q1=q1)

    def load_exp_curve(self, trim_tail=0):
        return self._curve


@pytest.fixture
def stub_pipeline(monkeypatch):
    def fake_sim(density, U_0, r0, n, m, outdir, **kw):
        b = resolve_table_bounds("modified_lj", U_0, n, m, r0, None, N=5000, density=density)
        return {"rmax": b["rmax"], "t_tol_lj": b["t_tol_lj"], "n_pairs_below_rmin": 3}

    def fake_saxs(save_dir, **kw):
        d = Path(save_dir) / "S(q)_data"
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "average_structure_factor.npy", crystal_curve(q1=0.0131, amp=6.0))

    monkeypatch.setattr(bo, "run_simulation", fake_sim)
    monkeypatch.setattr(bo, "convert_to_SAXS_fft", fake_saxs)


def _read_blocks(path):
    blocks, header = [], None
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if not row:
                header = None
            elif row[0].startswith("# Iteration"):
                blocks.append({"head": row, "rows": []})
            elif row[0] == "iteration":
                header = row
            elif header:
                blocks[-1]["rows"].append(dict(zip(header, row)))
    return blocks


def _objective(tmp_path, datasets, metric="shift_rmse"):
    ps = bo.ParamSpace({"global": {"n": {"bounds": (11.0, 13.0), "init": 12.0}}, "local": {}},
                       dataset_ids=[d.id for d in datasets])
    obj = bo.make_global_objective(
        datasets, ps, ffpath="", out_root=str(tmp_path), trim_tail=0, mode="sim",
        metric=metric, compare_q_range=W_NARROW,
    )
    return obj, ps


def test_audit_fields_from_sim_dict_and_done_strings(tmp_path):
    (tmp_path / "shift_rmse_diagnostics.json").write_text(json.dumps(
        {"failed": False, "m4": 0.25, "shift_term": 0.03, "loss": 0.28}))
    sim = {"rmax": 4.946506116169816, "t_tol_lj": 0.1 / 3.0, "n_pairs_below_rmin": 2}
    a = bo._audit_fields(sim, 2.5, str(tmp_path))
    b = bo._audit_fields({k: str(v) for k, v in sim.items()}, 2.5, str(tmp_path))
    assert a == b
    assert a["rmax_over_r0"] == pytest.approx(4.946506116169816 / 2.5)
    assert (a["rmse"], a["shift"], a["n_pairs_below_rmin"]) == (0.25, 0.03, 2)
    fixed = bo._audit_fields({"rmax": "6.0", "t_tol_lj": "None"}, 2.5, str(tmp_path / "x"))
    assert fixed["t_tol_lj"] == "" and fixed["rmse"] == "" and fixed["n_pairs_below_rmin"] == ""


def test_sequential_records_audit_per_dataset(tmp_path, stub_pipeline):
    datasets = [_DS("d0", 1.0, 0.0129), _DS("d1", 6.0, 0.0129)]
    obj, ps = _objective(tmp_path, datasets)
    total = float(obj(ps.init_unit(), ffpath=""))

    (block,) = _read_blocks(tmp_path / "bo_trajectory.csv")
    rows = {r["dataset_id"]: r for r in block["rows"]}
    for ds in datasets:
        r = rows[ds.id]
        b = resolve_table_bounds("modified_lj", ds.sim.U0, 12.0, 6.0, 2.5, None, N=5000, density=0.005)
        assert float(r["rmax"]) == pytest.approx(b["rmax"])
        assert float(r["rmax_over_r0"]) == pytest.approx(b["rmax"] / 2.5)
        assert float(r["t_tol_lj"]) == pytest.approx(0.1 / ds.sim.U0)
        assert r["n_pairs_below_rmin"] == "3"
        diag = json.loads((tmp_path / "eval_000" / ds.id / "shift_rmse_diagnostics.json").read_text())
        assert float(r["rmse"]) == pytest.approx(diag["m4"])
        assert float(r["shift"]) == pytest.approx(diag["shift_term"])
        assert float(r["loss"]) == pytest.approx(diag["loss"])
    assert rows["d0"]["rmax"] != rows["d1"]["rmax"]
    assert "mu_b" in block["rows"][0] and "sigma_b" in block["rows"][0]

    with open(tmp_path / "loss_components.txt", newline="") as fh:
        comp = list(csv.DictReader(fh))
    assert [c["dataset_id"] for c in comp] == ["d0", "d1"]
    for c in comp:
        assert float(c["rmse"]) + float(c["shift"]) == pytest.approx(float(c["loss"]))
        assert float(c["total_loss"]) == pytest.approx(total)
        assert c["iteration"] == "0"


def test_non_shift_rmse_leaves_components_blank(tmp_path, stub_pipeline):
    obj, ps = _objective(tmp_path, [_DS("d0", 3.0, 0.0129)], metric="mse")
    obj(ps.init_unit(), ffpath="")
    (block,) = _read_blocks(tmp_path / "bo_trajectory.csv")
    row = block["rows"][0]
    assert row["rmse"] == "" and row["shift"] == "" and row["rmax"] != ""


def test_warm_start_reads_old_and_new_blocks(tmp_path, stub_pipeline):
    traj = tmp_path / "bo_trajectory.csv"
    old_cols = ["iteration", "dataset_id", "loss", "k", "alpha", "A", "mu_c", "mu_b",
                "sigma_c", "sigma_b", "K_s", "density", "n", "m", "r0", "U0"]
    old = dict(zip(old_cols, [0, "d0", 0.5, "", "", "", "", "", "", "", "", 0.005, 11.5, 6.0, 2.5, 3.0]))
    bo._write_iteration_block(str(traj), 0, 0.5, [old])

    obj, ps = _objective(tmp_path, [_DS("d0", 3.0, 0.0129)])
    obj._eval_id = 1
    new_loss = float(obj(ps.init_unit(), ffpath=""))

    train_x, train_y, next_id = bo.load_warm_start_from_trajectory(traj, ps)
    assert next_id == 2
    assert train_x.shape == (2, 1)
    assert torch.allclose(ps.unit_to_phys(train_x[:, 0:1]).squeeze(-1),
                          torch.tensor([11.5, 12.0], dtype=torch.float64))
    assert train_y.squeeze(-1).tolist() == pytest.approx([-0.5, -new_loss])


def test_parallel_collector_records_audit(tmp_path):
    sim_dir = tmp_path / "eval_000" / "d0"
    sim_dir.mkdir(parents=True)
    result = {"rmax": "4.946506116169816", "t_tol_lj": "0.03333333333333333",
              "n_pairs_below_rmin": "5", "L": "100.0"}
    (sim_dir / "DONE").write_text(json.dumps({"loss": 0.28, "result": result}))
    (sim_dir / "shift_rmse_diagnostics.json").write_text(json.dumps(
        {"failed": False, "m4": 0.25, "shift_term": 0.03, "loss": 0.28}))
    job = sp.Job(idx=0, name="j", ds_id="d0", sim_dir=sim_dir, config_path=sim_dir / "c",
                 sbatch_path=sim_dir / "s", out_path=sim_dir / "o", done_status="DONE")
    plan = dict(ds=SimpleNamespace(id="d0", weight=1.0), density=0.005, n=12.0, m=6.0,
                r0=2.5, U0=3.0, save_dir=str(sim_dir))
    total, ok, (rec,) = bo._collect_parallel_eval_loss(
        eval_id=0, G={}, plans=[plan], finished_jobs=[job], out_root=str(tmp_path))
    assert ok and total == pytest.approx(0.28)
    assert rec["rmax"] == pytest.approx(4.946506116169816)
    assert rec["rmax_over_r0"] == pytest.approx(4.946506116169816 / 2.5)
    assert rec["t_tol_lj"] == pytest.approx(0.1 / 3.0)
    assert (rec["n_pairs_below_rmin"], rec["rmse"], rec["shift"]) == (5, 0.25, 0.03)

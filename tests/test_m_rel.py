"""m_rel parameterization of m in mode='sim': m = 4 + m_rel (n - 5)."""
import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import bo
from shift_rmse_curves import crystal_curve
from simulation import resolve_table_bounds


class _DS:
    def __init__(self, ds_id="d0"):
        self.id = ds_id
        self.weight = 1.0
        self.datatype = "sq"
        self.exp_path = f"/dev/null/{ds_id}.npy"
        self.sim = SimpleNamespace(n=12.0, m=6.0, density=0.005, r0=None, U0=None)

    def load_exp_curve(self, trim_tail=0):
        return crystal_curve(q1=0.0129)


@pytest.fixture
def stub_pipeline(monkeypatch):
    def fake_sim(density, U_0, r0, n, m, outdir, **kw):
        b = resolve_table_bounds(U_0, n, m, r0, N=5000, density=density)
        return {"rmax": b["rmax"], "t_tol": b["t_tol"], "n_pairs_below_rmin": 0}

    def fake_saxs(save_dir, **kw):
        d = Path(save_dir) / "S(q)_data"
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "average_structure_factor.npy", crystal_curve(q1=0.0131, amp=6.0))

    monkeypatch.setattr(bo, "run_simulation", fake_sim)
    monkeypatch.setattr(bo, "convert_to_SAXS_fft", fake_saxs)


def _cfg(**over):
    cfg = {
        "U0": {"bounds": (0.5, 8.0), "init": 3.0},
        "r0": {"bounds": (2.0, 3.5), "init": 2.5},
        "n": {"bounds": (5.5, 15.0), "init": 12.0},
        "m_rel": {"bounds": (0.0, 1.0), "init": 2.0 / 7.0},
        "density": {"fixed": 0.005},
    }
    cfg.update(over)
    return {k: v for k, v in cfg.items() if v is not None}


def _rows(path):
    rows, header = [], None
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if not row or row[0].startswith("# Iteration"):
                header = None if not row else header
            elif row[0] == "iteration":
                header = row
            elif header:
                rows.append(dict(zip(header, row)))
    return rows


@pytest.mark.parametrize("n", [5.5, 6.0, 12.0, 15.0])
@pytest.mark.parametrize("u", [0.0, 0.3, 2.0 / 7.0, 1.0])
def test_convention_matches_the_inverse_drivers(n, u):
    m = bo.m_from_m_rel(n, u)
    assert m == pytest.approx(4.0 + u * (n - 1.0 - 4.0))   # ParamSpaceConstrainedNM.decode
    assert bo.m_rel_from_m(n, m) == pytest.approx(u, abs=1e-12)
    assert 4.0 <= m <= n - 1.0


def test_driver_initial_point():
    # 2026-08/09 drivers: n = 12, m = 6 -> m_rel = (6 - 4) / (12 - 1 - 4) = 2/7
    assert bo.m_rel_from_m(12.0, 6.0) == pytest.approx(2.0 / 7.0)
    assert bo.m_from_m_rel(12.0, 2.0 / 7.0) == pytest.approx(6.0)


def test_n_at_or_below_5_rejected():
    with pytest.raises(ValueError, match="n > 5"):
        bo.m_from_m_rel(5.0, 0.5)
    with pytest.raises(ValueError, match="n > 5"):
        bo.m_rel_from_m(4.5, 4.0)


def test_resolve_sim_params_uses_m_rel():
    G = {"U0": 3.0, "r0": 2.5, "n": 10.0, "m_rel": 0.5, "density": 0.005}
    density, r0, U0, n, m = bo._resolve_sim_params(_DS(), G, "sim")
    assert (density, r0, U0, n) == (0.005, 2.5, 3.0, 10.0)
    assert m == pytest.approx(6.5)


@pytest.mark.parametrize("cfg, match", [
    (_cfg(m={"bounds": (4.0, 8.0)}), "m or m_rel"),
    (_cfg(n=None), "needs n"),
    (_cfg(n={"bounds": (5.0, 15.0)}), "n > 5"),
    (_cfg(n={"fixed": 5.0}), "n > 5"),
    (_cfg(m_rel={"bounds": (-0.1, 1.0)}), r"\[0, 1\]"),
    (_cfg(m_rel={"bounds": (0.0, 1.2)}), r"\[0, 1\]"),
    (_cfg(m_rel={"fixed": 1.5}), r"\[0, 1\]"),
])
def test_invalid_m_rel_spaces_rejected(cfg, match):
    with pytest.raises(ValueError, match=match):
        bo._validate_param_mode(bo.ParamSpace(cfg), "sim")


def test_fixed_n_and_fixed_m_rel_allowed():
    bo._validate_param_mode(bo.ParamSpace(_cfg(n={"fixed": 12.0})), "sim")
    bo._validate_param_mode(bo.ParamSpace(_cfg(m_rel={"fixed": 0.25})), "sim")


def test_m_rel_not_allowed_in_map_mode():
    with pytest.raises(ValueError, match="mode='map'"):
        bo._validate_param_mode(bo.ParamSpace({"m_rel": {"bounds": (0.0, 1.0)}}), "map")


def test_trajectory_column_and_warm_start(tmp_path, stub_pipeline):
    ps = bo.ParamSpace(_cfg())
    obj = bo.make_global_objective(
        [_DS()], ps, ffpath="", out_root=str(tmp_path), trim_tail=0, mode="sim",
        metric="shift_rmse", compare_q_range=None,
    )
    xs = [ps.init_unit(), torch.tensor([0.9, 0.1, 0.05, 0.83], dtype=torch.float64)]
    losses = [float(obj(x, ffpath="")) for x in xs]

    rows = _rows(tmp_path / "bo_trajectory.csv")
    assert len(rows) == 2
    for x, r in zip(xs, rows):
        G = ps.decode(ps.unit_to_phys(x))
        assert float(r["m_rel"]) == pytest.approx(G["m_rel"])
        assert float(r["n"]) == pytest.approx(G["n"])
        assert float(r["m"]) == pytest.approx(4.0 + G["m_rel"] * (G["n"] - 5.0))
    with open(tmp_path / "eval_001" / "d0" / "sim_params_d0.csv", newline="") as fh:
        sp_row = next(csv.DictReader(fh))
    assert float(sp_row["m_rel"]) == pytest.approx(float(rows[1]["m_rel"]))

    train_x, train_y, next_id = bo.load_warm_start_from_trajectory(
        tmp_path / "bo_trajectory.csv", ps)
    assert next_id == 2
    assert torch.allclose(train_x, torch.stack(xs), atol=1e-12)
    assert train_y.squeeze(-1).tolist() == pytest.approx([-v for v in losses])


def test_m_rel_column_blank_without_m_rel(tmp_path, stub_pipeline):
    ps = bo.ParamSpace({"n": {"bounds": (11.0, 13.0), "init": 12.0},
                        "U0": {"fixed": 3.0}, "r0": {"fixed": 2.5}})
    obj = bo.make_global_objective(
        [_DS()], ps, ffpath="", out_root=str(tmp_path), trim_tail=0, mode="sim",
        metric="shift_rmse", compare_q_range=None,
    )
    obj(ps.init_unit(), ffpath="")
    (row,) = _rows(tmp_path / "bo_trajectory.csv")
    assert row["m_rel"] == "" and row["m"] == "6.0"


def _old_driver_block(path, iteration, n, m, extra=None):
    rec = {"iteration": iteration, "dataset_id": "d0", "loss": 0.3, "k": "", "alpha": "",
           "density": 0.005, "n": n, "m": m, "r0": 2.5, "U0": 3.0, **(extra or {})}
    bo._write_iteration_block(str(path), iteration, 0.3, [rec])


def test_warm_start_reconstructs_m_rel_without_the_column(tmp_path):
    traj = tmp_path / "bo_trajectory.csv"
    _old_driver_block(traj, 0, 12.0, 6.0)
    _old_driver_block(traj, 1, 7.0, 5.0)
    ps = bo.ParamSpace(_cfg())
    train_x, _, _ = bo.load_warm_start_from_trajectory(traj, ps)
    m_rel = ps.unit_to_phys(train_x)[:, ps._names.index("m_rel")]
    assert m_rel.tolist() == pytest.approx([2.0 / 7.0, 0.5])


def test_warm_start_rejects_inconsistent_m_rel(tmp_path):
    traj = tmp_path / "bo_trajectory.csv"
    _old_driver_block(traj, 0, 12.0, 6.0, extra={"m_rel": 0.5})
    with pytest.raises(ValueError, match="m_rel=0.5"):
        bo.load_warm_start_from_trajectory(traj, bo.ParamSpace(_cfg()))

"""Physics-based mapping (task 0c-1): formulas, R3.2 search box, map mode in bo.py."""
import csv
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import bo
from datasets import (
    FIXED_LINKER_NM, PHYSICS_COEFFS, Dataset, ExperimentalParams, SimulationParams,
    check_mapped_params, physics_map,
)
from shift_rmse_curves import crystal_curve
from simulation import resolve_table_bounds

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "campaigns" / "r3.2-physics-validate"))
import train_config as r32  # noqa: E402  (re-exports the R3.1 ground truth)


def _ds(ds_id, L_bridge, C_chol, C_NaCl, exp_path="/dev/null/x.npy", density=0.005):
    return Dataset(
        id=ds_id, exp_path=exp_path,
        exp=ExperimentalParams(L_bridge=L_bridge, C_chol=C_chol, C_NaCl=C_NaCl),
        sim=SimulationParams(density=density), datatype="sq",
    )


# ---------------- formulas ---------------- #

def test_fixed_linker_length():
    assert FIXED_LINKER_NM == pytest.approx(32.04)


@pytest.mark.parametrize("k", [0.4, 0.65, 0.9])
@pytest.mark.parametrize("L_bridge", [20.0, 40.0, 80.0, 120.0])
def test_r0_matches_r0_sigma_at_defaults(k, L_bridge):
    ds = _ds("d0", L_bridge, 100.0, 10.0)
    p = ds.physics_params(k=k, A=1.0, K_s=0.0, a_m=3.0, delta=3.0)
    assert p["r0"] == pytest.approx(ds.r0_sigma(k=k), rel=1e-12)


def test_physics_map_values():
    p = physics_map(80.0, 140.0, 40.0, k=0.65, A=1.2, K_s=0.07, a_m=3.2, delta=3.2)
    r0 = 1 + 0.65 * (32.04 + 0.34 * 80.0) / 24.6
    g = r0 / (r0 - 1)
    assert p["r0"] == pytest.approx(r0)
    assert p["U0"] == pytest.approx(1.2 * 1.4 * (1 + 0.07 * 40.0))
    assert p["m"] == pytest.approx(3.2 * g)
    assert p["n"] == pytest.approx(6.4 * g)
    assert p["U0"] == pytest.approx(6.384)
    assert (p["n"], p["m"]) == (pytest.approx(10.49, abs=0.01), pytest.approx(5.24, abs=0.01))


def test_salt_only_raises_U0():
    lo = physics_map(20.0, 60.0, 0.0, k=0.65, A=1.2, K_s=0.07, a_m=3.2, delta=3.2)["U0"]
    hi = physics_map(20.0, 60.0, 40.0, k=0.65, A=1.2, K_s=0.07, a_m=3.2, delta=3.2)["U0"]
    assert lo == pytest.approx(0.72) and hi > lo


def test_check_mapped_params():
    check_mapped_params(0.5, 5.5, 4.0)
    for bad in [(0.49, 12.0, 6.0), (2.0, 5.4, 4.0), (2.0, 12.0, 3.9), (2.0, 6.0, 5.5)]:
        with pytest.raises(ValueError, match="outside the checked range"):
            check_mapped_params(*bad)
    with pytest.raises(ValueError, match="k must be > 0"):
        physics_map(20.0, 60.0, 0.0, k=0.0, A=1.0, K_s=0.0, a_m=3.0, delta=3.0)


# ---------------- R3.2 configuration ---------------- #

def test_r32_condition_set():
    assert len(r32.TRAIN) == 24 and len(r32.HELDOUT) == 5
    ids = [c["id"] for c in r32.ALL_CONDITIONS]
    assert len(set(ids)) == len(ids)
    train = {(c["L_bridge"], c["C_chol"], c["C_NaCl"]) for c in r32.TRAIN}
    assert not train & {(c["L_bridge"], c["C_chol"], c["C_NaCl"]) for c in r32.HELDOUT}
    assert all(c["id"][1:].isdigit() for c in r32.ALL_CONDITIONS)  # launcher job names


def test_r32_ground_truth_and_init_inside_box():
    g = r32.PARAM_CFG["global"]
    assert set(g) == set(PHYSICS_COEFFS)
    for name, spec in g.items():
        lo, hi = spec["bounds"]
        assert lo < r32.GROUND_TRUTH[name] < hi, name
        assert lo <= spec["init"] <= hi, name
        assert spec["init"] != pytest.approx(r32.GROUND_TRUTH[name]), name


def test_r32_ground_truth_inside_inverse_box():
    for c in r32.ALL_CONDITIONS:
        p = physics_map(c["L_bridge"], c["C_chol"], c["C_NaCl"], **r32.GROUND_TRUTH)
        assert 0.5 <= p["U0"] <= 8.0 and 5.5 <= p["n"] <= 15.0
        assert 4.0 <= p["m"] <= p["n"] - 1.0 and 2.0 <= p["r0"] <= 3.5


def test_r32_box_keeps_every_condition_in_range():
    """Grid over the 5-D search box x all conditions: limits hold, rmax < L/2, rmax <= 15."""
    g = r32.PARAM_CFG["global"]
    axes = [np.linspace(*g[name]["bounds"], 6) for name in PHYSICS_COEFFS]
    worst_rmax = 0.0
    for values in itertools.product(*axes):
        coeffs = dict(zip(PHYSICS_COEFFS, values))
        for c in r32.ALL_CONDITIONS:
            p = physics_map(c["L_bridge"], c["C_chol"], c["C_NaCl"], **coeffs)
            check_mapped_params(p["U0"], p["n"], p["m"])
            b = resolve_table_bounds(p["U0"], p["n"], p["m"], p["r0"], N=r32.N, density=r32.DENSITY)
            worst_rmax = max(worst_rmax, b["rmax"])
    assert worst_rmax < 15.0


# ---------------- bo.py map mode ---------------- #

def _ps(dataset_ids, cfg=None):
    return bo.ParamSpace(cfg or r32.PARAM_CFG, dataset_ids=dataset_ids)


def test_map_mode_rejects_old_and_incomplete_spaces():
    base = dict(r32.PARAM_CFG["global"])
    bad_cfgs = [
        {**base, "alpha": {"bounds": (1.0, 5.0), "init": 3.0}},
        {**base, "n": {"bounds": (6.0, 20.0), "init": 12.0}},
        {k: v for k, v in base.items() if k != "delta"},
    ]
    for cfg in bad_cfgs:
        with pytest.raises(ValueError, match="mode='map'"):
            bo._validate_param_mode(_ps(["d0"], {"global": cfg, "local": {}}), "map")
    with pytest.raises(ValueError, match="mode='map'"):
        bo._validate_param_mode(
            _ps(["d0"], {"global": base, "local": {"U0": {"bounds": (1, 2), "init": 1.5}}}), "map")
    with pytest.raises(ValueError, match="mode='sim'"):
        bo._validate_param_mode(_ps(["d0"], {"global": {"k": {"fixed": 0.6}}, "local": {}}), "sim")


def test_resolve_map_params_and_density():
    ds = _ds("d0", 80.0, 140.0, 40.0)
    out = bo._resolve_sim_params(ds, dict(r32.GROUND_TRUTH), {"d0": {}}, "map")
    p = ds.physics_params(**r32.GROUND_TRUTH)
    assert out == (0.005, p["r0"], p["U0"], p["n"], p["m"])
    ds.sim.density = None
    with pytest.raises(ValueError, match="density"):
        bo._resolve_sim_params(ds, dict(r32.GROUND_TRUTH), {"d0": {}}, "map")
    low = {**r32.GROUND_TRUTH, "A": 0.5}
    with pytest.raises(ValueError, match="U0"):
        bo._resolve_sim_params(_ds("d0", 20.0, 60.0, 0.0), low, {"d0": {}}, "map")


def _prepare(datasets, G, tmp_path):
    return bo._parallel_prepare_eval_jobs(
        datasets=datasets, eval_id=0, G=G, L={d.id: {} for d in datasets},
        out_root=str(tmp_path), trim_tail=0, sim_defaults={"N": 5000},
        mode="map", scattering_method="saxsfft", scattering_kwargs={},
        metric="shift_rmse", compare_q_range=None, dp_coeff=0.5, plot_apdist=False,
        ffpath="", metric_kwargs={},
    )


def test_parallel_prepare_map_mode(tmp_path):
    datasets = [_ds("d0", 20.0, 60.0, 0.0), _ds("d23", 80.0, 140.0, 40.0)]
    specs, plans, fails, reason = _prepare(datasets, dict(r32.GROUND_TRUTH), tmp_path)
    assert reason is None and not fails
    for spec, ds in zip(specs, datasets):
        p = ds.physics_params(**r32.GROUND_TRUTH)
        rk = spec["worker_config"]["run_kwargs"]
        assert (rk["density"], rk["U_0"], rk["r0"], rk["n"], rk["m"], rk["N"]) == pytest.approx(
            (0.005, p["U0"], p["r0"], p["n"], p["m"], 5000))
        assert spec["worker_config"]["loss"]["compare_q_range"] is None
        with open(tmp_path / "eval_000" / ds.id / f"sim_params_{ds.id}.csv", newline="") as fh:
            row = next(csv.DictReader(fh))
        assert float(row["a_m"]) == pytest.approx(3.2) and "alpha" not in row
    specs, plans, fails, reason = _prepare(datasets, {**r32.GROUND_TRUTH, "A": 0.5}, tmp_path / "x")
    assert reason is not None and specs == [] and len(fails) == 1  # d0 maps to U0 = 0.3


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


def test_map_mode_trajectory_and_warm_start(tmp_path, stub_pipeline):
    datasets = []
    for c in (r32.TRAIN[0], r32.TRAIN[-1]):
        path = tmp_path / f"{c['id']}.npy"
        np.save(path, crystal_curve(q1=0.0129))
        datasets.append(_ds(c["id"], c["L_bridge"], c["C_chol"], c["C_NaCl"], exp_path=path))
    ps = _ps([d.id for d in datasets])
    out = tmp_path / "run"
    out.mkdir()
    obj = bo.make_global_objective(
        datasets, ps, ffpath="", out_root=str(out), trim_tail=0, sim_defaults={"N": 5000},
        mode="map", metric="shift_rmse", compare_q_range=None,
    )
    xs = [ps.init_unit(), ps.phys_to_unit(torch.tensor(
        [r32.GROUND_TRUTH[n] for n in ps._names], dtype=torch.float64))]
    losses = [float(obj(x, ffpath="")) for x in xs]

    rows = []
    with open(out / "bo_trajectory.csv", newline="") as fh:
        header = None
        for row in csv.reader(fh):
            if not row or row[0].startswith("# Iteration"):
                header = None if not row else header
                continue
            if row[0] == "iteration":
                header = row
            elif header:
                rows.append(dict(zip(header, row)))
    assert len(rows) == 4
    for r in rows:
        assert all(r[c] != "" for c in PHYSICS_COEFFS)
        ds = next(d for d in datasets if d.id == r["dataset_id"])
        coeffs = {c: float(r[c]) for c in PHYSICS_COEFFS}
        p = ds.physics_params(**coeffs)
        assert float(r["U0"]) == pytest.approx(p["U0"]) and float(r["n"]) == pytest.approx(p["n"])

    train_x, train_y, next_id = bo.load_warm_start_from_trajectory(out / "bo_trajectory.csv", ps)
    assert next_id == 2
    assert torch.allclose(train_x, torch.stack([x.reshape(-1) for x in xs]), atol=1e-9)
    assert train_y.squeeze(-1).tolist() == pytest.approx([-v for v in losses])

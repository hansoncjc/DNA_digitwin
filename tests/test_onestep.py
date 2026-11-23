"""One-step inverse design (R7): BO over experimental conditions through a
mapping file (bo mode="map" with conditions in the ParamSpace; inverse.py
CONFIG "mapping"). HOOMD and botorch stubbed as in test_inverse."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import bo
import inverse
import mapping
from datasets import CONDITION_NAMES, Dataset, ExperimentalParams, SimulationParams, physics_map
from test_inverse import _write_config, pipeline, target  # noqa: F401  (fixtures)

REPO = Path(__file__).resolve().parents[1]
R71 = REPO / "campaigns" / "inverse" / "r7.1-onestep-fcc0.05" / "config.py"
EVAL56_JSON = REPO / "campaigns" / "r3.2-physics-validate" / "mapping" / "r3.2_eval056.json"
COEFFS = {"k": 0.6520927389752293, "A": 1.2028501815438974, "K_s": 0.07251128398087511,
          "a_m": 3.2087563613999377, "delta": 3.3893817640184665}


def _ds(**exp):
    return Dataset(id="d0", exp_path="unused.npy", exp=ExperimentalParams(**exp),
                   sim=SimulationParams(density=0.005))


def _map_cfg(**conds):
    cfg = {k: {"fixed": v} for k, v in COEFFS.items()}
    cfg.update(conds)
    return cfg


# ---------------- bo: mode="map" with conditions ---------------- #

def test_map_mode_accepts_conditions_and_still_needs_every_coefficient():
    bo._validate_param_mode(bo.ParamSpace(_map_cfg()), "map")
    bo._validate_param_mode(bo.ParamSpace(_map_cfg(
        L_bridge={"bounds": (20.0, 180.0)}, C_NaCl={"fixed": 20.0})), "map")
    with pytest.raises(ValueError, match="Not allowed: \\['r0'\\]"):
        bo._validate_param_mode(bo.ParamSpace(_map_cfg(r0={"bounds": (2.0, 3.0)})), "map")
    bad = _map_cfg(L_bridge={"bounds": (20.0, 180.0)})
    del bad["delta"]
    with pytest.raises(ValueError, match="missing: \\['delta'\\]"):
        bo._validate_param_mode(bo.ParamSpace(bad), "map")
    with pytest.raises(ValueError, match="mode='sim'"):
        bo._validate_param_mode(bo.ParamSpace({"L_bridge": {"bounds": (20.0, 180.0)}}), "sim")


def test_conditions_from_the_paramspace_override_the_dataset():
    ds = _ds(L_bridge=20.0, C_anchor=60.0, C_NaCl=0.0)
    G = {**COEFFS, "L_bridge": 120.0, "C_NaCl": 30.0}     # C_anchor from the dataset
    density, r0, U0, n, m = bo._resolve_sim_params(ds, G, "map")
    p = physics_map(120.0, 60.0, 30.0, **COEFFS)
    assert (density, r0, U0, n, m) == (0.005, p["r0"], p["U0"], p["n"], p["m"])


@pytest.mark.parametrize("x", [(20.0, 60.0, 0.0), (80.0, 140.0, 40.0), (40.0, 100.0, 20.0)])
def test_without_conditions_map_mode_is_what_it_was(x):
    """Training (R3.2): conditions from dataset.exp; bitwise the old
    dataset.physics_params path."""
    ds = _ds(L_bridge=x[0], C_anchor=x[1], C_NaCl=x[2])
    got = bo._resolve_sim_params(ds, dict(COEFFS), "map")
    p = ds.physics_params(**COEFFS)
    assert got == (0.005, p["r0"], p["U0"], p["n"], p["m"])


def test_map_mode_without_bridge_still_fails():
    with pytest.raises(ValueError, match="L_bridge is required"):
        bo._resolve_sim_params(_ds(), dict(COEFFS), "map")


def test_condition_columns_only_when_the_paramspace_holds_them():
    assert bo._condition_columns(dict(COEFFS)) == {}
    assert bo._condition_columns({"U0": 1.0, "n": 12.0}) == {}
    G = {**COEFFS, "C_NaCl": 20, "L_bridge": 80.0}
    assert list(bo._condition_columns(G).items()) == [("L_bridge", 80.0), ("C_NaCl", 20.0)]


def test_failed_block_columns(tmp_path):
    path = tmp_path / "bo_trajectory.csv"
    rec = bo._failed_trajectory_record(0, "d0", dict(COEFFS))
    bo._write_failed_iteration_block(str(path), 0, [rec], reason="x")
    header = path.read_text().splitlines()[0].split(",")
    assert tuple(header) == bo._TRAJECTORY_PARAM_COLUMNS          # unchanged without conditions

    path2 = tmp_path / "b.csv"
    G = {**COEFFS, "L_bridge": 80.0, "C_anchor": 100.0, "C_NaCl": 20.0}
    rec = bo._failed_trajectory_record(0, "d0", G)
    bo._write_failed_iteration_block(str(path2), 0, [rec], reason="x")
    rows = list(csv.reader(path2.open()))
    assert rows[0][-3:] == list(CONDITION_NAMES)
    assert rows[2][-3:] == ["80.0", "100.0", "20.0"]


# ---------------- inverse.py with a mapping file ---------------- #

def _mapping_file(tmp_path, coeffs=COEFFS):
    return mapping.save_mapping(tmp_path / "mapping.json", coeffs).path


SEARCH = {
    "L_bridge": {"bounds": (20.0, 180.0), "init": 80.0},
    "C_anchor": {"bounds": (45.0, 170.0), "init": 100.0},
    "C_NaCl": {"bounds": (0.0, 40.0), "init": 20.0},
}


def _onestep_config(tmp_path, target, **over):
    return _write_config(tmp_path, target, mapping=over.pop("mapping", None)
                         or _mapping_file(tmp_path), search=over.pop("search", SEARCH), **over)


def test_search_must_match_the_mode(tmp_path, target):
    sim_search = inverse.load_config(_write_config(tmp_path, target))["search"]
    with pytest.raises(inverse.ConfigError, match="exactly its conditions"):
        inverse.run(_onestep_config(tmp_path, target, search=sim_search), check=True)
    with pytest.raises(inverse.ConfigError, match="search must hold U0"):
        inverse.run(_write_config(tmp_path, target, search=SEARCH), check=True)
    with pytest.raises(inverse.ConfigError, match="mapping: mapping file not found"):
        inverse.run(_onestep_config(tmp_path, target, mapping=str(tmp_path / "no.json")),
                    check=True)


def test_box_point_outside_the_checked_range(tmp_path, pipeline, target):  # noqa: F811
    search = {**SEARCH, "C_anchor": {"bounds": (10.0, 170.0), "init": 100.0}}  # U0 < 0.5
    with pytest.raises(inverse.ConfigError, match="U0=.* < 0.5"):
        inverse.run(_onestep_config(tmp_path, target, search=search), check=True)
    assert pipeline["calls"] == 0


def test_check_maps_the_grid(tmp_path, pipeline, target):  # noqa: F811
    out = inverse.run(_onestep_config(tmp_path, target), check=True)
    rmax = out["info"]["rmax"]
    assert rmax["grid_points"] == 7 ** 3 and rmax["rmax"] < rmax["L_half"]
    assert set(rmax["at"]) == set(CONDITION_NAMES) and set(rmax["mapped"]) == {"r0", "U0", "n", "m"}
    assert out["info"]["mapping"]["coefficients"] == COEFFS
    assert pipeline["calls"] == 0


def test_onestep_run_resume_and_summary(tmp_path, pipeline, target):  # noqa: F811
    path = _onestep_config(tmp_path, target, max_evals=2)
    summary = inverse.run(path)
    out = tmp_path / "out"
    snap = json.loads((out / "inverse_config.json").read_text())
    assert snap["locked"]["mapping"]["coefficients"] == COEFFS
    assert snap["locked"]["mapping"]["sha256"] == mapping.file_sha256(tmp_path / "mapping.json")
    assert snap["mapping"]["path"] == str((tmp_path / "mapping.json").resolve())

    # Every evaluation simulated what the mapping gives at its conditions.
    with open(out / "bo_trajectory.csv", newline="") as fh:
        rows = [r for r in csv.reader(fh)]
    header = next(r for r in rows if r and r[0] == "iteration")
    assert header[header.index("m_rel") + 1:header.index("m_rel") + 4] == list(CONDITION_NAMES)
    data = [dict(zip(header, r)) for r in rows if r and r[0].isdigit()]
    assert len(data) == 2
    for rec in data:
        p = physics_map(*(float(rec[k]) for k in CONDITION_NAMES), **COEFFS)
        assert {k: float(rec[k]) for k in ("r0", "U0", "n", "m")} == pytest.approx(p, rel=1e-12)
        assert {k: float(rec[k]) for k in COEFFS} == COEFFS
    first = data[0]
    assert [float(first[k]) for k in CONDITION_NAMES] == pytest.approx([80.0, 100.0, 20.0])
    sim_params = next((out / "eval_000" / "d0").glob("sim_params_*.csv")).read_text()
    assert sim_params.splitlines()[0].endswith("m_rel,L_bridge,C_anchor,C_NaCl")
    best = summary["best"]
    assert set(CONDITION_NAMES) <= set(best) and summary["mapping"].endswith("mapping.json")
    assert "conditions: L_bridge" in (out / "inverse_summary.txt").read_text()

    # Resume: the conditions are read back from the trajectory.
    summary = inverse.run(_onestep_config(tmp_path, target, max_evals=3))
    assert pipeline["run_bo_log"][-1]["warm_start"][0].shape == (2, 3)
    assert summary["n_successful"] == 3


def test_changed_mapping_is_refused_on_resume(tmp_path, pipeline, target):  # noqa: F811
    inverse.run(_onestep_config(tmp_path, target, max_evals=1))
    other = mapping.save_mapping(tmp_path / "other" / "mapping.json", {**COEFFS, "k": 0.66})
    calls = pipeline["calls"]
    with pytest.raises(inverse.ConfigError, match="mapping: run started with"):
        inverse.run(_onestep_config(tmp_path, target, max_evals=2, mapping=other.path))
    assert pipeline["calls"] == calls
    # The same content at another path resumes.
    moved = tmp_path / "moved.json"
    moved.write_bytes((tmp_path / "mapping.json").read_bytes())
    assert inverse.run(_onestep_config(tmp_path, target, max_evals=2,
                                       mapping=str(moved)))["n_successful"] == 2


def test_sim_runs_are_what_they_were(tmp_path, pipeline, target):  # noqa: F811
    """mapping None: mode="sim", no condition columns, locked mapping None,
    summary without mapping keys."""
    summary = inverse.run(_write_config(tmp_path, target, max_evals=1))
    out = tmp_path / "out"
    snap = json.loads((out / "inverse_config.json").read_text())
    assert snap["locked"]["mapping"] is None and "mapping" not in snap
    assert list(snap["locked"]) == list(inverse.LOCKED)
    header = next(r for r in csv.reader((out / "bo_trajectory.csv").open())
                  if r and r[0] == "iteration")
    assert not set(CONDITION_NAMES) & set(header)
    assert "mapping" not in summary and not set(CONDITION_NAMES) & set(summary["best"])


def test_snapshot_without_mapping_resumes(tmp_path, pipeline, target):  # noqa: F811
    """inverse_config.json written before "mapping" existed (R6 runs)."""
    inverse.run(_write_config(tmp_path, target, max_evals=1))
    snap_path = tmp_path / "out" / "inverse_config.json"
    snap = json.loads(snap_path.read_text())
    del snap["locked"]["mapping"]
    snap_path.write_text(json.dumps(snap))
    assert inverse.run(_write_config(tmp_path, target, max_evals=2))["n_successful"] == 2


@pytest.mark.parametrize("name", ["r6.4-fcc0.10", "r6.5-fcc0.05", "r6.6-hs-fluid",
                                  "r6.8a-fcc0.10-lam0", "r6.8b-hs-lam0", "r6.8c-hs-lam2"])
def test_r6_configs_normalize_as_before(name):
    """The R6 configs: the only new field is mapping = None, locked as None."""
    path = REPO / "campaigns" / "inverse" / name / "config.py"
    cfg = inverse.normalize_config(inverse.load_config(path))
    assert cfg["mapping"] is None and inverse.run_mode(cfg) == "sim"
    cfg["target_md5"] = "0" * 32                          # set by preflight
    locked = inverse.locked_fields(cfg)
    assert locked["mapping"] is None and list(locked) == list(inverse.LOCKED)


# ---------------- the R7.1 campaign config ---------------- #

@pytest.fixture(scope="module")
def r71():
    return inverse.load_config(R71)


def test_r71_uses_the_eval56_mapping_and_the_r65_target(r71):
    m = mapping.load_mapping(r71["mapping"])
    assert m.coefficients == COEFFS and Path(r71["mapping"]) == EVAL56_JSON
    assert hashlib.md5(Path(r71["target"]).read_bytes()).hexdigest() == \
        "780fd41a9f25b73dba663037a2415a25"
    r65 = inverse.load_config(REPO / "campaigns" / "inverse" / "r6.5-fcc0.05" / "config.py")
    for key in ("target", "N", "density", "max_evals", "batch_size", "bo_seed", "potential",
                "metric", "scattering"):
        assert r71[key] == r65[key], key
    assert "metric_kwargs" not in r71                     # shift_rmse defaults, lambda 1
    assert r71["parallel_cfg"]["partition"] == "gpu-a40"
    assert r71["parallel_cfg"]["time"] == "04:00:00"
    cfg = inverse.normalize_config(r71)
    assert inverse.run_mode(cfg) == "map"


def test_r71_box_is_the_image_rule(r71):
    m = mapping.load_mapping(r71["mapping"])
    box = {k: r71["search"][k]["bounds"] for k in CONDITION_NAMES}
    assert box["C_NaCl"] == (0.0, 40.0)
    lo = m.apply({k: box[k][0] for k in CONDITION_NAMES})
    hi = m.apply({k: box[k][1] for k in CONDITION_NAMES})
    assert lo["r0"] == pytest.approx(2.0, abs=1e-12) and hi["r0"] == pytest.approx(3.5, abs=1e-12)
    assert lo["U0"] == pytest.approx(0.5, abs=1e-12) and hi["U0"] == pytest.approx(8.0, abs=1e-12)
    # Numbers quoted in the docstring and the plan.
    assert box["L_bridge"] == pytest.approx((16.72, 183.15), abs=5e-3)
    assert box["C_anchor"] == pytest.approx((41.57, 170.52), abs=5e-3)
    # Every point of the box maps inside the R6.5 search box (all four parameters).
    r65 = inverse.load_config(REPO / "campaigns" / "inverse" / "r6.5-fcc0.05" / "config.py")
    grid = np.meshgrid(*[np.linspace(*box[k], 9) for k in CONDITION_NAMES], indexing="ij")
    for x in zip(*(g.ravel() for g in grid)):
        p = m.apply(dict(zip(CONDITION_NAMES, x)))
        p["m_rel"] = bo.m_rel_from_m(p["n"], p["m"])
        for k in ("U0", "r0", "n", "m_rel"):
            a, b = r65["search"][k]["bounds"]
            assert a - 1e-9 <= p[k] <= b + 1e-9, (x, k, p[k])


def test_r71_initial_point(r71):
    m = mapping.load_mapping(r71["mapping"])
    x0 = {k: r71["search"][k]["init"] for k in CONDITION_NAMES}
    assert x0 == {"L_bridge": 80.0, "C_anchor": 100.0, "C_NaCl": 20.0}
    p = m.apply(x0)
    assert p["r0"] == pytest.approx(2.5703, abs=1e-4) and p["U0"] == pytest.approx(2.9473, abs=1e-4)

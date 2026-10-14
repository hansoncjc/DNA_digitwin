"""R5.1 prediction set: coefficients from R3.2 eval 56, conditions, checks before submission."""
import csv
import sys
from pathlib import Path

import pytest

from datasets import check_mapped_params, physics_map
from simulation import resolve_table_bounds

CAMPAIGNS = Path(__file__).resolve().parents[1] / "campaigns"
sys.path.insert(0, str(CAMPAIGNS / "r5.1-predict"))
import predict_config as r51  # noqa: E402
import make_prediction  # noqa: E402

GT_KEYS = {(c["L_bridge"], c["C_anchor"], c["C_NaCl"]) for c in r51.GT_CONDITIONS}


def test_coefficients_are_eval_56():
    rows = list(csv.DictReader(open(CAMPAIGNS / "r3.2-physics-validate" / "analysis" / "r32_evals.csv")))
    e56 = next(r for r in rows if r["eval"] == "56")
    for name, value in r51.COEFFS.items():
        assert value == pytest.approx(float(e56[name]), rel=1e-5)


def test_condition_set():
    keys = [(c["L_bridge"], c["C_anchor"], c["C_NaCl"]) for c in r51.CONDITIONS]
    assert len(keys) == len(set(keys)) == 14
    assert not set(keys) & GT_KEYS
    assert [c["id"] for c in r51.CONDITIONS] == [f"p{i}" for i in range(14)]
    assert sum(c["part"] == "part1" for c in r51.CONDITIONS) == 8
    # part 1 is the extended C_anchor x C_NaCl grid at L_bridge = 80 minus the training cells
    part1 = {k for k, c in zip(keys, r51.CONDITIONS) if c["part"] == "part1"}
    assert part1 == {(80.0, c, s) for c in (45.0, 60.0, 100.0, 140.0) for s in (0.0, 10.0, 20.0, 40.0, 60.0)
                     if c == 45.0 or s == 60.0}
    # part 2 leaves out the ground-truth cells d4-d7, d16-d19, h0, h3
    part2 = {k for k, c in zip(keys, r51.CONDITIONS) if c["part"] == "part2"}
    assert part2 == {(L, 100.0, s) for L in (40.0, 120.0) for s in (0.0, 10.0, 40.0)}


@pytest.mark.parametrize("cond, tag", [
    ((80.0, 45.0, 0.0), "extrap(C_anchor)"),
    ((80.0, 45.0, 60.0), "extrap(C_anchor,C_NaCl)"),
    ((80.0, 100.0, 60.0), "extrap(C_NaCl)"),
    ((40.0, 100.0, 10.0), "interp"),
    ((120.0, 100.0, 0.0), "extrap(L_bridge)"),
])
def test_training_range_tag(cond, tag):
    assert r51.training_range_tag(*cond) == tag


@pytest.mark.parametrize("cond", r51.CONDITIONS, ids=lambda c: c["id"])
def test_every_condition_inside_checked_range(cond):
    p = physics_map(cond["L_bridge"], cond["C_anchor"], cond["C_NaCl"], **r51.COEFFS)
    check_mapped_params(p["U0"], p["n"], p["m"])
    b = resolve_table_bounds(p["U0"], p["n"], p["m"], p["r0"], N=r51.N, density=r51.DENSITY)
    assert b["rmin"] < b["rmax"] < b["L"] / 2


def test_list_reports_no_failure(capsys):
    assert make_prediction.list_conditions() == 0
    assert "14 conditions, 0 outside the checked range" in capsys.readouterr().out


def test_mapped_matches_r32_trajectory_at_d0():
    # eval 56, d0 (20, 60, 0) as logged in the R3.2 bo_trajectory.csv and job log i56_d00.out
    p, b = make_prediction.mapped({"L_bridge": 20.0, "C_anchor": 60.0, "C_NaCl": 0.0})
    assert p["r0"] == 2.0295643082031667
    assert p["U0"] == 0.7217101089263385
    assert p["n"] == 13.006808349169374
    assert p["m"] == 6.325372133560858
    assert b["rmax"] == 3.082059002946405
    assert b["t_tol"] == 0.13855978842913302
    assert round(b["rmin"], 4) == 1.0148

"""mapping.py: mapping files (save, load, checks, export from a training trajectory)."""
import csv
import json
from pathlib import Path

import pytest

import bo
import mapping
from datasets import PHYSICS_COEFFS, physics_map

REPO = Path(__file__).resolve().parents[1]
EVAL56_CSV = REPO / "campaigns" / "r6.10-translation" / "inputs" / "r3.2_eval_056.csv"
EVAL56_JSON = REPO / "campaigns" / "r3.2-physics-validate" / "mapping" / "r3.2_eval056.json"
GT_JSON = REPO / "campaigns" / "r3.1-gt" / "mapping" / "ground_truth.json"
COEFFS = {"k": 0.6, "A": 1.3, "K_s": 0.05, "a_m": 3.1, "delta": 3.4}
CONDITIONS = [(20.0, 60.0, 0.0), (80.0, 100.0, 20.0), (164.7, 242.6, 37.5)]


def _csv_rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_round_trip_is_exact(tmp_path):
    path = tmp_path / "m.json"
    saved = mapping.save_mapping(path, COEFFS, source={"run": "test"})
    loaded = mapping.load_mapping(path)
    assert loaded.coefficients == COEFFS and saved.sha256 == loaded.sha256
    assert loaded.sha256 == mapping.file_sha256(path)
    assert loaded.condition_names == ("L_bridge", "C_anchor", "C_NaCl")
    assert loaded.constants == mapping.physics_constants()
    assert loaded.source == {"run": "test"}


@pytest.mark.parametrize("x", CONDITIONS)
def test_apply_is_physics_map(tmp_path, x):
    m = mapping.save_mapping(tmp_path / "m.json", COEFFS)
    got = m.apply(dict(zip(m.condition_names, x)))
    assert got == physics_map(*x, **COEFFS)            # bitwise


@pytest.mark.parametrize("edit, match", [
    (lambda d: d.update(format="other"), "format"),
    (lambda d: d.update(version=2), "format"),
    (lambda d: d.update(form="nn"), "form"),
    (lambda d: d["coefficients"].pop("delta"), "coefficients"),
    (lambda d: d["coefficients"].update(extra=1.0), "coefficients"),
    (lambda d: d["coefficients"].update(k="0.6"), "finite number"),
    (lambda d: d["coefficients"].update(k=True), "finite number"),
    (lambda d: d.update(conditions=["C_anchor", "L_bridge", "C_NaCl"]), "conditions"),
    (lambda d: d.update(outputs=["r0", "U0"]), "outputs"),
    (lambda d: d["constants"].update(fixed_linker_nm=30.0), "differ from the code"),
    (lambda d: d["constants"].pop("c_anchor_ref"), "differ from the code"),
    (lambda d: d.pop("constants"), "constants"),
])
def test_malformed_files_are_refused(tmp_path, edit, match):
    path = tmp_path / "m.json"
    data = mapping.save_mapping(path, COEFFS).to_dict()
    edit(data)
    path.write_text(json.dumps(data))
    with pytest.raises(mapping.MappingError, match=match):
        mapping.load_mapping(path)


def test_missing_and_non_json_files(tmp_path):
    with pytest.raises(mapping.MappingError, match="not found"):
        mapping.load_mapping(tmp_path / "nope.json")
    (tmp_path / "bad.json").write_text("{")
    with pytest.raises(mapping.MappingError, match="not JSON"):
        mapping.load_mapping(tmp_path / "bad.json")


def test_apply_needs_every_condition(tmp_path):
    m = mapping.save_mapping(tmp_path / "m.json", COEFFS)
    with pytest.raises(mapping.MappingError, match="C_NaCl"):
        m.apply({"L_bridge": 20.0, "C_anchor": 100.0})


def test_export_from_a_block_trajectory(tmp_path):
    """A bo_trajectory.csv as mode="map" writes it (blocks with own headers)."""
    traj = tmp_path / "bo_trajectory.csv"
    for it, c in ((0, {**COEFFS, "k": 0.5}), (1, COEFFS)):
        rows = [{"iteration": it, "dataset_id": f"d{i}", "loss": 0.1, **c, "density": 0.005,
                 "n": 12.0, "m": 6.0, "r0": 2.5, "U0": 3.0, "m_rel": ""} for i in range(3)]
        bo._write_iteration_block(str(traj), it, 0.3, rows)
    m = mapping.export_from_trajectory(traj, 1, tmp_path / "m.json", run="test")
    assert m.coefficients == COEFFS
    assert m.source["eval_id"] == 1 and m.source["rows"] == 3
    assert m.source["sha256"] == mapping.file_sha256(traj)
    with pytest.raises(mapping.MappingError, match="no rows of evaluation 7"):
        mapping.export_from_trajectory(traj, 7, tmp_path / "x.json")


def test_export_refuses_inconsistent_rows(tmp_path):
    traj = tmp_path / "rows.csv"
    rows = [{"iteration": 3, "dataset_id": f"d{i}", **COEFFS, "k": 0.6 + i} for i in range(2)]
    with open(traj, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(mapping.MappingError, match="differ"):
        mapping.coefficients_from_trajectory(traj, 3)


def test_cli_export(tmp_path):
    out = tmp_path / "m.json"
    assert mapping.main(["export", str(EVAL56_CSV), "56", str(out), "--run", "R3.2"]) == 0
    assert mapping.load_mapping(out).coefficients == mapping.load_mapping(EVAL56_JSON).coefficients


# ---------------- files in the repository ---------------- #

def test_r32_eval56_file_holds_the_eval56_coefficients():
    m = mapping.load_mapping(EVAL56_JSON)
    rows = _csv_rows(EVAL56_CSV)
    assert len(rows) == 24 and {r["iteration"] for r in rows} == {"56"}
    assert m.coefficients == {k: float(rows[0][k]) for k in PHYSICS_COEFFS}
    prov = json.loads((EVAL56_CSV.parent / "provenance.json").read_text())
    assert m.source["sha256"] == prov["r3.2_eval_056.csv"]["sha256"]
    assert m.source["sha256"] == mapping.file_sha256(EVAL56_CSV)
    assert m.source["eval_id"] == 56 and m.source["run"] == "R3.2"


def test_r32_eval56_file_reproduces_the_simulated_parameters():
    """The 24 R3.2 eval-56 rows record what was simulated; the file maps the
    same conditions to the same (r0, U0, n, m)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "gt_config", REPO / "campaigns" / "r3.1-gt" / "gt_config.py")
    gt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gt)
    m = mapping.load_mapping(EVAL56_JSON)
    by_id = {c["id"]: c for c in gt.TRAIN}
    for row in _csv_rows(EVAL56_CSV):
        c = by_id[row["dataset_id"]]
        p = m.apply(c)
        for k in ("r0", "U0", "n", "m"):
            assert p[k] == pytest.approx(float(row[k]), rel=1e-12), (row["dataset_id"], k)


def test_ground_truth_file_holds_gt_config():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "gt_config", REPO / "campaigns" / "r3.1-gt" / "gt_config.py")
    gt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gt)
    m = mapping.load_mapping(GT_JSON)
    assert m.coefficients == {k: float(v) for k, v in gt.GROUND_TRUTH.items()}
    assert m.source["sha256"] == mapping.file_sha256(REPO / "campaigns" / "r3.1-gt" / "gt_config.py")

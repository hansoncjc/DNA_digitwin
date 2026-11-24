"""
Mapping files: a trained (or ground-truth) mapping from experimental
conditions x to simulation parameters, stored as one JSON file.

A mapping file is the hand-over from mapping training (R3) to every use of
the trained mapping (one-step inverse design R7, prediction, translation).
Only the physics-based form exists so far; another form (e.g. a neural
network) is a new entry in ``FORMS`` with its own ``apply``.

File layout (``FORMAT``, ``VERSION`` 1)::

    {
      "format": "digitwin-mapping", "version": 1,
      "form": "physics",
      "conditions": ["L_bridge", "C_anchor", "C_NaCl"],
      "outputs": ["r0", "U0", "n", "m"],
      "coefficients": {"k": ..., "A": ..., "K_s": ..., "a_m": ..., "delta": ...},
      "constants": {"d_si": 24.6, "fixed_linker_nm": 32.04,
                    "dsdna_rise_nm": 0.34, "c_anchor_ref": 100.0},
      "source": {...}            # where the coefficients come from (free form)
    }

``constants`` records the fixed values of ``datasets.physics_map``; a file
whose constants differ from the code is refused, so a file cannot describe a
mapping other than the one the code computes. Floats are written with
``json`` (repr), so a save/load round trip is exact.

    python mapping.py export TRAJECTORY EVAL_ID OUT --run R3.2
        coefficients of one evaluation of a mapping-training bo_trajectory.csv
        (or a CSV holding only its rows) written as a mapping file
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from datasets import (  # noqa: E402
    C_ANCHOR_REF, CONDITION_NAMES, DSDNA_RISE_NM, FIXED_LINKER_NM, PHYSICS_COEFFS,
    physics_map,
)

FORMAT = "digitwin-mapping"
VERSION = 1
OUTPUTS = ("r0", "U0", "n", "m")
D_SI_DEFAULT = 24.6   # ExperimentalParams.d_si default (sigma, nm)

#: form -> coefficient names and the conditions it reads.
FORMS = {
    "physics": {"coefficients": PHYSICS_COEFFS, "conditions": CONDITION_NAMES},
}


class MappingError(ValueError):
    """A mapping file is missing, malformed or does not match the code."""


def physics_constants(d_si=D_SI_DEFAULT):
    """The fixed values of ``datasets.physics_map`` (d_si is the particle diameter)."""
    return {
        "d_si": float(d_si),
        "fixed_linker_nm": float(FIXED_LINKER_NM),
        "dsdna_rise_nm": float(DSDNA_RISE_NM),
        "c_anchor_ref": float(C_ANCHOR_REF),
    }


class Mapping:
    """A loaded mapping: ``apply`` maps one condition to (r0, U0, n, m)."""

    def __init__(self, form, coefficients, constants, source=None, path=None, sha256=None):
        self.form = form
        self.coefficients = dict(coefficients)
        self.constants = dict(constants)
        self.source = dict(source or {})
        self.path = path
        self.sha256 = sha256
        self.condition_names = tuple(FORMS[form]["conditions"])
        self.outputs = OUTPUTS

    def apply(self, condition):
        """``{r0, U0, n, m}`` at ``condition`` ({name: value} over ``condition_names``)."""
        missing = [k for k in self.condition_names if k not in condition]
        if missing:
            raise MappingError(f"condition is missing {missing}")
        x = [float(condition[k]) for k in self.condition_names]
        return physics_map(*x, **self.coefficients, d_si=self.constants["d_si"])

    def to_dict(self):
        return {
            "format": FORMAT,
            "version": VERSION,
            "form": self.form,
            "conditions": list(self.condition_names),
            "outputs": list(self.outputs),
            "coefficients": {k: float(self.coefficients[k]) for k in FORMS[self.form]["coefficients"]},
            "constants": dict(self.constants),
            "source": dict(self.source),
        }

    def locked(self):
        """What identifies this mapping in a run snapshot (not its path)."""
        d = self.to_dict()
        d.pop("source")
        d["sha256"] = self.sha256
        return d


def physics_condition_box(mapping, r0_range, u0_range, salt_range, l_bridge_min):
    """
    Search box in x for one-step inverse design with a physics mapping (R7).

    Rule (author, 2026-11-22; floor 2026-11-23): for the given salt range, the
    largest box in x whose image lies inside ``r0_range`` x ``u0_range``, with
    L_bridge at least ``l_bridge_min`` (a bridge cannot be absent):

    - L_bridge: r0 = 1 + k (l + rise L_bridge) / d_si inverted at the two ends
      of ``r0_range``; the lower end is raised to ``l_bridge_min`` when the
      inversion gives less (then the box reaches r0 only down to
      r0(l_bridge_min));
    - C_anchor: U0 = A (C_anchor / c_ref)(1 + K_s C_NaCl) grows with C_anchor
      and salt, so the lower end puts U0 = u0_range[0] at the lowest salt and
      the upper end U0 = u0_range[1] at the highest salt.

    Returns {name: (lo, hi)} over ``CONDITION_NAMES``.
    """
    if mapping.form != "physics":
        raise MappingError(f"physics_condition_box needs a physics mapping, not {mapping.form!r}")
    c, k = mapping.coefficients, mapping.constants
    if not (c["k"] > 0 and c["A"] > 0 and c["K_s"] >= 0):
        raise MappingError(f"physics_condition_box needs k > 0, A > 0, K_s >= 0: {c}")

    def bridge(r0):
        return (k["d_si"] * (r0 - 1.0) / c["k"] - k["fixed_linker_nm"]) / k["dsdna_rise_nm"]

    def anchor(U0, s):
        return k["c_anchor_ref"] * U0 / (c["A"] * (1.0 + c["K_s"] * s))

    box = {
        "L_bridge": (max(float(l_bridge_min), bridge(r0_range[0])), bridge(r0_range[1])),
        "C_anchor": (anchor(u0_range[0], salt_range[0]), anchor(u0_range[1], salt_range[1])),
        "C_NaCl": (float(salt_range[0]), float(salt_range[1])),
    }
    for name, (lo, hi) in box.items():
        if not lo < hi:
            raise MappingError(f"empty box in {name}: ({lo}, {hi})")
    return box


def _check(data, where):
    if not isinstance(data, dict):
        raise MappingError(f"{where}: expected a JSON object")
    if data.get("format") != FORMAT or data.get("version") != VERSION:
        raise MappingError(
            f"{where}: format/version {data.get('format')!r}/{data.get('version')!r}; "
            f"expected {FORMAT!r}/{VERSION}"
        )
    form = data.get("form")
    if form not in FORMS:
        raise MappingError(f"{where}: form {form!r}; available: {sorted(FORMS)}")
    spec = FORMS[form]
    if list(data.get("conditions", [])) != list(spec["conditions"]):
        raise MappingError(f"{where}: conditions {data.get('conditions')!r}; "
                           f"form {form!r} reads {list(spec['conditions'])}")
    if list(data.get("outputs", [])) != list(OUTPUTS):
        raise MappingError(f"{where}: outputs {data.get('outputs')!r}; expected {list(OUTPUTS)}")
    coeffs = data.get("coefficients")
    if not isinstance(coeffs, dict) or set(coeffs) != set(spec["coefficients"]):
        raise MappingError(f"{where}: coefficients {sorted(coeffs or {})}; "
                           f"form {form!r} needs exactly {list(spec['coefficients'])}")
    for k, v in coeffs.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
            raise MappingError(f"{where}: coefficient {k}={v!r} is not a finite number")
    const = data.get("constants")
    if not isinstance(const, dict) or "d_si" not in const:
        raise MappingError(f"{where}: constants must hold d_si and the physics_map constants")
    expected = physics_constants(const["d_si"])
    if const != expected:
        raise MappingError(
            f"{where}: constants {const} differ from the code's physics_map constants "
            f"{expected}; this file describes another mapping"
        )


def load_mapping(path):
    """Read and check a mapping file; returns a :class:`Mapping`."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise MappingError(f"mapping file not found: {path}")
    raw = path.read_bytes()
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise MappingError(f"{path}: not JSON: {exc}") from exc
    _check(data, str(path))
    return Mapping(data["form"], data["coefficients"], data["constants"],
                   source=data.get("source"), path=str(path),
                   sha256=hashlib.sha256(raw).hexdigest())


def save_mapping(path, coefficients, *, form="physics", source=None, d_si=D_SI_DEFAULT):
    """Write a mapping file (checked as ``load_mapping`` checks it); returns the loaded Mapping."""
    m = Mapping(form, {k: float(v) for k, v in coefficients.items()}, physics_constants(d_si),
                source=source)
    data = m.to_dict()
    _check(data, str(path))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return load_mapping(path)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def coefficients_from_trajectory(trajectory, eval_id):
    """
    Coefficients of evaluation ``eval_id`` from a mapping-training
    ``bo_trajectory.csv`` (block format) or a plain CSV of its rows. Every
    row of that evaluation must carry the same coefficients.
    """
    header = None
    rows = []
    with open(trajectory, newline="") as fh:
        for row in csv.reader(fh):
            if not row:
                continue
            if row[0] == "iteration" and len(row) > 1 and row[1] == "dataset_id":
                header = row
                continue
            if header is None or row[0] != str(int(eval_id)) or len(row) < len(header):
                continue
            rows.append(dict(zip(header, row)))
    if not rows:
        raise MappingError(f"{trajectory}: no rows of evaluation {eval_id}")
    try:
        coeffs = [{k: float(r[k]) for k in PHYSICS_COEFFS} for r in rows]
    except (KeyError, ValueError) as exc:
        raise MappingError(f"{trajectory}: evaluation {eval_id} has no numeric "
                           f"{list(PHYSICS_COEFFS)}: {exc}") from exc
    if any(c != coeffs[0] for c in coeffs):
        raise MappingError(f"{trajectory}: evaluation {eval_id} rows differ in coefficients")
    return coeffs[0], len(rows)


def export_from_trajectory(trajectory, eval_id, out, *, run=None, note=None):
    """Write the coefficients of one training evaluation as a mapping file."""
    coeffs, n_rows = coefficients_from_trajectory(trajectory, eval_id)
    traj = Path(trajectory).resolve()
    try:
        shown = traj.relative_to(REPO).as_posix()
    except ValueError:
        shown = str(traj)
    source = {"run": run, "eval_id": int(eval_id), "file": shown,
              "sha256": file_sha256(traj), "rows": n_rows}
    if note:
        source["note"] = note
    return save_mapping(out, coeffs, source=source)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export", help="one training evaluation -> mapping file")
    ex.add_argument("trajectory")
    ex.add_argument("eval_id", type=int)
    ex.add_argument("out")
    ex.add_argument("--run", help="training run label, e.g. R3.2")
    ex.add_argument("--note")
    a = ap.parse_args(argv)
    m = export_from_trajectory(a.trajectory, a.eval_id, a.out, run=a.run, note=a.note)
    print(f"[mapping] wrote {a.out} (sha256 {m.sha256}): {m.coefficients}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

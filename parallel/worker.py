"""
Per-job worker for parallel BO evaluations.

Reads a JSON config describing one dataset's evaluation, runs the full
pipeline end-to-end on a single GPU node:

    simulation.run_simulation -> scattering.convert_to_SAXS[_fft]
    -> metrics.compare_to_exp[_saxsfft]

and writes a `DONE` flag (JSON) into `outdir` containing the loss plus
diagnostic metadata. If any stage raises, a `FAILED` flag is written
with the traceback instead. The launcher polls these flag files.

Usage (called from inside a SLURM job script)::

    python worker.py <config.json>

Expected JSON config layout::

    {
      "outdir": "/abs/path/sim_dir",
      "run_kwargs": {
          "density": 0.05, "U_0": 25.0, "r0": 2.2, "n": 12.0, "m": 6.0,
          "N": 5000, "steps": 15000000, "device": "gpu",
          "potential": "modified_lj", "seed": 42, "plot": false
      },
      "scattering": {
          "method": "saxsfft",
          "kwargs": {"N_grid": 300}
      },
      "loss": {
          "exp_path":         "/abs/.../average_structure_factor.npy",
          "trim_tail":        0,
          "datatype":         "sq",
          "ffpath":           "/abs/.../sasmodels_sphere_fit.txt",
          "metric":           "mse",
          "scattering_method":"saxsfft",
          "q_min":            0.02,
          "q_max":            0.03
      }
    }

Status flag files written in <outdir>::

    RUNNING  - created at start (pid/host/start time)
    DONE     - created on success (JSON with loss, run_time_seconds, result)
    FAILED   - created on any exception (run_time_seconds prefix + traceback)

Python: written for the HOOMD venv (3.9+).
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# Make sure DNA_digitwin/ is importable even if PYTHONPATH is unset for some
# reason (sbatch scripts normally export it, but this is a safety net).
_THIS_DIR = Path(__file__).resolve().parent          # .../DNA_digitwin/parallel
_PARENT   = _THIS_DIR.parent                         # .../DNA_digitwin
for p in (_PARENT, _THIS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from simulation import run_simulation                            # noqa: E402
from scattering  import convert_to_SAXS, convert_to_SAXS_fft, extract_exp_sq  # noqa: E402
from metrics     import compare_to_exp, compare_to_exp_saxsfft  # noqa: E402


# ---------------------------------------------------------------------------
# Flag helpers
# ---------------------------------------------------------------------------

def _write_flag(outdir: Path, name: str, text: str = "") -> None:
    (outdir / name).write_text(text)


def _clear_flags(outdir: Path) -> None:
    for name in ("RUNNING", "DONE", "FAILED"):
        p = outdir / name
        if p.exists():
            p.unlink()


# ---------------------------------------------------------------------------
# Pipeline helpers (mirror the sequential body in bo.make_global_objective)
# ---------------------------------------------------------------------------

_SQ_CANDIDATES = (
    "S(q)_data/average_structure_factor.npy",
    "scattering_data/average_structure_factor.npy",
    "S(q)_/average_structure_factor.npy",
    "S(q)/average_structure_factor.npy",
)


def _locate_sim_sq(save_dir: Path) -> Path:
    for rel in _SQ_CANDIDATES:
        p = save_dir / rel
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Missing simulated S(q); tried {[str(save_dir / r) for r in _SQ_CANDIDATES]}"
    )


def _load_exp_curve(exp_path: str, trim_tail: int) -> np.ndarray:
    """
    Same semantics as Dataset.load_exp_curve (but without needing a
    Dataset instance inside the worker).
    """
    if trim_tail < 0:
        raise ValueError("trim_tail must be >= 0")
    arr = np.load(exp_path)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Expected (N, 2+) array in {exp_path}, got {arr.shape}")
    arr = arr[:, :2].astype(np.float64, copy=False)
    if trim_tail and arr.shape[0] > trim_tail:
        arr = arr[:-trim_tail]
    return arr


def _run_pipeline(cfg: dict, save_dir: Path) -> dict:
    """
    Execute sim -> SAXS conversion -> compare-to-exp. Returns a dict
    {"loss": float, "sim_result": dict} on success; raises on failure.
    """
    # 1) Simulation
    sim_result = run_simulation(outdir=str(save_dir), **cfg["run_kwargs"])

    # 2) Sim -> S(q)
    scat = cfg.get("scattering", {}) or {}
    method = scat.get("method", "saxsfft")
    sc_kw  = dict(scat.get("kwargs", {}) or {})
    if method == "saxsfft":
        convert_to_SAXS_fft(str(save_dir), **sc_kw)
    else:
        convert_to_SAXS(str(save_dir), **sc_kw)

    # 3) Compare to experiment
    loss_cfg     = cfg["loss"]
    sim_sq_path  = _locate_sim_sq(save_dir)
    sim_sq       = np.load(sim_sq_path)

    exp_data = _load_exp_curve(loss_cfg["exp_path"],
                               int(loss_cfg.get("trim_tail", 0)))
    datatype = loss_cfg.get("datatype", "sq")
    if datatype == "sq":
        exp_sq = exp_data
    else:
        exp_sq = extract_exp_sq(
            exp_scattering=exp_data,
            ffpath=loss_cfg["ffpath"],
            q_min=float(loss_cfg.get("q_min", 0.02)),
            q_max=float(loss_cfg.get("q_max", 0.03)),
            normalize=False,
        )

    compare_method = loss_cfg.get("scattering_method", method)
    metric         = loss_cfg.get("metric", "mse")
    compare_q_range = loss_cfg.get("compare_q_range", (0.003, 0.06))
    if compare_method == "saxsfft":
        loss = float(compare_to_exp_saxsfft(
            exp_sq,
            sim_sq,
            str(save_dir),
            metric=metric,
            q_range=compare_q_range,
        ))
    else:
        loss = float(compare_to_exp(exp_sq, sim_sq, str(save_dir), metric=metric))

    return {"loss": loss, "sim_result": sim_result}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv):
    if len(argv) != 2:
        print(f"usage: {argv[0]} <config.json>", file=sys.stderr)
        return 2

    cfg    = json.loads(Path(argv[1]).read_text())
    outdir = Path(cfg["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)

    host = os.uname().nodename
    _clear_flags(outdir)
    _write_flag(
        outdir,
        "RUNNING",
        f"pid={os.getpid()}\nhost={host}\nstart={time.time()}\n",
    )

    t0 = time.time()
    try:
        out = _run_pipeline(cfg, outdir)
        dt  = time.time() - t0

        sim_result = out["sim_result"]
        try:
            stringified = {k: str(v) for k, v in sim_result.items()}
        except Exception:
            stringified = {"_repr": str(sim_result)}

        summary = {
            "run_time_seconds": dt,
            "host":             host,
            "loss":             float(out["loss"]),
            "result":           stringified,
        }
        _write_flag(outdir, "DONE", json.dumps(summary, indent=2))
        (outdir / "RUNNING").unlink(missing_ok=True)
        print(f"[worker] SUCCESS loss={out['loss']:.6g} in {dt:.1f}s -> {outdir}")
        return 0
    except Exception:
        dt = time.time() - t0
        tb = traceback.format_exc()
        _write_flag(
            outdir,
            "FAILED",
            f"run_time_seconds={dt:.1f}\nhost={host}\n\n{tb}",
        )
        (outdir / "RUNNING").unlink(missing_ok=True)
        print(f"[worker] FAILED after {dt:.1f}s -> {outdir}\n{tb}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))

"""
Per-job worker for parallel BO evaluations.

Reads a JSON config describing one dataset's evaluation, runs the full
pipeline end-to-end on a single GPU node:

    simulation.run_simulation -> scattering.convert_to_SAXS[_fft]
    -> metrics.compare_to_exp[_saxsfft]

and writes a `DONE` flag (JSON) into `outdir` containing the loss plus
diagnostic metadata. If the metric raises `metrics.MetricFailed` (the
S(q) exists but no loss can be computed from it), a `METRIC_FAILED` flag
is written; the launcher does not resubmit it. Any other exception writes
a `FAILED` flag with the traceback, which the launcher may resubmit. The
launcher polls these flag files.

Usage (called from inside a SLURM job script)::

    python worker.py <config.json>

Expected JSON config layout::

    {
      "outdir": "/abs/path/sim_dir",
      "owner":  "i37_d00:<hex>",   # written by the launcher (prepare_jobs)
      "run_kwargs": {
          "density": 0.005,  # number density N/V in particles/σ³ (NOT volume fraction)
          "U_0": 3.0, "r0": 2.5, "n": 12.0, "m": 6.0,
          "N": 5000, "potential": "mie_r0", "plot": false
          # any other run_simulation keyword (steps, device, seed, t_tol,
          # rmax, ...) is passed through; the forward model leaves them at
          # their defaults (README "Forward model")
      },
      "scattering": {
          "method": "saxsfft",
          "kwargs": {"N_grid": 600}
      },
      "loss": {
          "exp_path":         "/abs/.../average_structure_factor.npy",
          "trim_tail":        0,
          "datatype":         "sq",
          "ffpath":           "/abs/.../sasmodels_sphere_fit.txt",
          "metric":           "mse",
          "scattering_method":"saxsfft",
          "q_min":            0.02,
          "q_max":            0.03,
          "metric_kwargs":    {}       # shift_rmse parameters (metrics.shift_rmse_params)
      }
    }

Folder ownership: the launcher writes ``<outdir>/OWNER`` with the token it
puts in ``owner``. The worker checks it before it writes anything, after the
simulation, and before the final flag; when OWNER holds another token (this
job was given up and a newer one took the folder) it exits with status 3 and
writes nothing more. The S(q) is computed from the GSD this run's simulation
wrote (``run_simulation``'s ``gsd_path``), never from another GSD in the folder.

Status flag files written in <outdir>::

    RUNNING        - created at start (pid/host/start time)
    DONE           - created on success (JSON with loss, run_time_seconds, result)
    METRIC_FAILED  - metrics.MetricFailed (JSON with reason, run_time_seconds,
                     host, result); not retried
    FAILED         - created on any other exception (run_time_seconds prefix
                     + traceback)

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
from metrics     import MetricFailed, compare_to_exp, compare_to_exp_saxsfft  # noqa: E402


# ---------------------------------------------------------------------------
# Flag helpers
# ---------------------------------------------------------------------------

OWNER_FILE = "OWNER"     # same name as parallel.submit_parallel.OWNER_FILE
NOT_OWNER_EXIT = 3


class NotOwner(Exception):
    """The folder now belongs to another job; this one must not write there."""


def _owner_of(outdir: Path):
    try:
        return (outdir / OWNER_FILE).read_text().strip()
    except (FileNotFoundError, NotADirectoryError):
        return None


def _check_owner(outdir: Path, token: str) -> None:
    current = _owner_of(outdir)
    if current != token:
        raise NotOwner(f"{outdir} belongs to {current!r}, not to this job ({token!r})")


def _write_flag(outdir: Path, name: str, text: str = "") -> None:
    (outdir / name).write_text(text)


def _write_final_flag(outdir: Path, token: str, name: str, text: str) -> None:
    """Write DONE / METRIC_FAILED / FAILED and remove RUNNING, if still the owner."""
    _check_owner(outdir, token)
    _write_flag(outdir, name, text)
    (outdir / "RUNNING").unlink(missing_ok=True)


def _clear_flags(outdir: Path) -> None:
    for name in ("RUNNING", "DONE", "METRIC_FAILED", "FAILED"):
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


def _run_pipeline(cfg: dict, save_dir: Path, state: dict, owner: str) -> dict:
    """
    Execute sim -> SAXS conversion -> compare-to-exp. Returns a dict
    {"loss": float, "sim_result": dict} on success; raises on failure.
    ``state["sim_result"]`` is set as soon as the simulation finishes, so
    the caller can report it when a later stage fails. Raises ``NotOwner``
    before the S(q) is written if the folder changed hands meanwhile.
    """
    scat = cfg.get("scattering", {}) or {}
    method = scat.get("method", "saxsfft")
    sc_kw  = dict(scat.get("kwargs", {}) or {})
    if "path" in sc_kw:
        raise ValueError("scattering kwargs must not set 'path': the S(q) is "
                         "computed from the GSD of this run's simulation")

    # 1) Simulation
    sim_result = run_simulation(outdir=str(save_dir), **cfg["run_kwargs"])
    state["sim_result"] = sim_result
    _check_owner(save_dir, owner)

    # 2) Sim -> S(q), from this run's own GSD
    gsd_path = str(sim_result["gsd_path"])
    if method == "saxsfft":
        convert_to_SAXS_fft(str(save_dir), path=gsd_path, **sc_kw)
    else:
        convert_to_SAXS(str(save_dir), path=gsd_path, **sc_kw)

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
    dp_coeff       = loss_cfg.get("dp_coeff", 0.5)
    plot_apdist    = loss_cfg.get("plot_apdist", True)
    metric_kwargs  = loss_cfg.get("metric_kwargs") or None
    if compare_method == "saxsfft":
        loss = float(compare_to_exp_saxsfft(
            exp_sq,
            sim_sq,
            str(save_dir),
            metric=metric,
            q_range=compare_q_range,
            dp_coeff=dp_coeff,
            plot_apdist=plot_apdist,
            metric_kwargs=metric_kwargs,
        ))
    else:
        loss = float(compare_to_exp(
            exp_sq,
            sim_sq,
            str(save_dir),
            metric=metric,
            dp_coeff=dp_coeff,
            plot_apdist=plot_apdist,
        ))

    return {"loss": loss, "sim_result": sim_result}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _stringify(sim_result) -> dict:
    try:
        return {k: str(v) for k, v in sim_result.items()}
    except Exception:
        return {"_repr": str(sim_result)}


def main(argv):
    if len(argv) != 2:
        print(f"usage: {argv[0]} <config.json>", file=sys.stderr)
        return 2

    cfg    = json.loads(Path(argv[1]).read_text())
    outdir = Path(cfg["outdir"])
    owner  = cfg.get("owner")
    if not owner:
        print("[worker] config has no 'owner' (written by parallel.prepare_jobs)",
              file=sys.stderr)
        return 2
    try:
        return _run(cfg, outdir, owner)
    except NotOwner as exc:
        print(f"[worker] {exc}; exiting without writing into it", file=sys.stderr)
        return NOT_OWNER_EXIT


def _run(cfg: dict, outdir: Path, owner: str) -> int:
    # Before anything is written: an older job that starts after a newer one
    # took the folder must not clear the newer job's flags.
    _check_owner(outdir, owner)
    host = os.uname().nodename
    _clear_flags(outdir)
    _write_flag(
        outdir,
        "RUNNING",
        f"pid={os.getpid()}\nhost={host}\nstart={time.time()}\n",
    )

    t0 = time.time()
    state: dict = {}
    try:
        out = _run_pipeline(cfg, outdir, state, owner)
        dt  = time.time() - t0

        summary = {
            "run_time_seconds": dt,
            "host":             host,
            "loss":             float(out["loss"]),
            "result":           _stringify(out["sim_result"]),
        }
        _write_final_flag(outdir, owner, "DONE", json.dumps(summary, indent=2))
        print(f"[worker] SUCCESS loss={out['loss']:.6g} in {dt:.1f}s -> {outdir}")
        return 0
    except NotOwner:
        raise
    except MetricFailed as exc:
        dt = time.time() - t0
        summary = {
            "run_time_seconds": dt,
            "host":             host,
            "reason":           str(exc),
            "result":           _stringify(state.get("sim_result", {})),
        }
        _write_final_flag(outdir, owner, "METRIC_FAILED", json.dumps(summary, indent=2))
        print(f"[worker] METRIC_FAILED after {dt:.1f}s -> {outdir}: {exc}", file=sys.stderr)
        return 0
    except Exception:
        dt = time.time() - t0
        tb = traceback.format_exc()
        _write_final_flag(
            outdir,
            owner,
            "FAILED",
            f"run_time_seconds={dt:.1f}\nhost={host}\n\n{tb}",
        )
        print(f"[worker] FAILED after {dt:.1f}s -> {outdir}\n{tb}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))

"""
Per-job worker for parallel BO evaluations.

Reads a JSON config describing one dataset's evaluation, then runs the
pipeline in isolated subprocess stages:

    simulation_stage.py -> analysis_stage.py

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

import subprocess
from collections import deque

_THIS_DIR = Path(__file__).resolve().parent          # .../DNA_digitwin/parallel
_PARENT   = _THIS_DIR.parent                         # .../DNA_digitwin
for p in (_PARENT, _THIS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

SIM_STAGE = _THIS_DIR / "simulation_stage.py"
ANALYSIS_STAGE = _THIS_DIR / "analysis_stage.py"


# ---------------------------------------------------------------------------
# Flag helpers
# ---------------------------------------------------------------------------

def _write_flag(outdir: Path, name: str, text: str = "") -> None:
    (outdir / name).write_text(text)


def _clear_flags(outdir: Path) -> None:
    for name in (
        "RUNNING",
        "DONE",
        "FAILED",
        "SIM_DONE.json",
        "SIM_FAILED.json",
        "ANALYSIS_DONE.json",
        "ANALYSIS_FAILED.json",
    ):
        p = outdir / name
        if p.exists():
            p.unlink()


# ---------------------------------------------------------------------------
# Stage orchestration
# ---------------------------------------------------------------------------

def _run_stage(name: str, args: list[str], *, tail_lines: int = 200) -> tuple[int, str]:
    """
    Run a subprocess while streaming its merged stdout/stderr to our stdout.

    Returns the exit code plus a bounded output tail for FAILED diagnostics.
    """
    print(f"[worker] starting {name}: {' '.join(args)}", flush=True)
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    tail = deque(maxlen=tail_lines)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        tail.append(line)
    rc = proc.wait()
    print(f"[worker] {name} exit_code={rc}", flush=True)
    return rc, "".join(tail)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_stage_failure(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        data = _read_json(path)
        return str(data.get("traceback", json.dumps(data, indent=2)))
    except Exception:
        return path.read_text(errors="replace")


def _format_failure(
    *,
    dt: float,
    host: str,
    stage: str,
    rc: int,
    tail: str,
    detail: str = "",
) -> str:
    parts = [
        f"run_time_seconds={dt:.1f}",
        f"host={host}",
        f"stage={stage}",
        f"exit_code={rc}",
        "",
    ]
    if detail:
        parts.extend(["stage_detail:", detail, ""])
    if tail:
        parts.extend(["subprocess_output_tail:", tail])
    return "\n".join(parts)


def _run_pipeline(config_path: Path, outdir: Path) -> dict:
    sim_done_path = outdir / "SIM_DONE.json"
    analysis_done_path = outdir / "ANALYSIS_DONE.json"

    sim_rc, sim_tail = _run_stage(
        "simulation_stage",
        [sys.executable, str(SIM_STAGE), str(config_path)],
    )
    if sim_rc != 0 and not sim_done_path.exists():
        detail = _load_stage_failure(outdir / "SIM_FAILED.json")
        raise RuntimeError(_format_failure(
            dt=0.0,
            host=os.uname().nodename,
            stage="simulation_stage",
            rc=sim_rc,
            tail=sim_tail,
            detail=detail,
        ))
    if sim_rc != 0 and sim_done_path.exists():
        print(
            "[worker] simulation_stage returned nonzero after writing "
            "SIM_DONE.json; continuing with analysis",
            flush=True,
        )

    analysis_rc, analysis_tail = _run_stage(
        "analysis_stage",
        [sys.executable, str(ANALYSIS_STAGE), str(config_path), str(sim_done_path)],
    )
    if analysis_rc != 0 or not analysis_done_path.exists():
        detail = _load_stage_failure(outdir / "ANALYSIS_FAILED.json")
        raise RuntimeError(_format_failure(
            dt=0.0,
            host=os.uname().nodename,
            stage="analysis_stage",
            rc=analysis_rc,
            tail=analysis_tail,
            detail=detail,
        ))

    sim_done = _read_json(sim_done_path)
    analysis_done = _read_json(analysis_done_path)
    return {
        "loss": float(analysis_done["loss"]),
        "sim_result": sim_done.get("result", {}),
        "sim_stage": sim_done,
        "analysis_stage": analysis_done,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv):
    if len(argv) != 2:
        print(f"usage: {argv[0]} <config.json>", file=sys.stderr)
        return 2

    config_path = Path(argv[1])
    cfg    = json.loads(config_path.read_text())
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
        out = _run_pipeline(config_path, outdir)
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
            "simulation_stage":  out.get("sim_stage", {}),
            "analysis_stage":    out.get("analysis_stage", {}),
        }
        _write_flag(outdir, "DONE", json.dumps(summary, indent=2))
        (outdir / "RUNNING").unlink(missing_ok=True)
        print(f"[worker] SUCCESS loss={out['loss']:.6g} in {dt:.1f}s -> {outdir}")
        return 0
    except Exception:
        dt = time.time() - t0
        tb = traceback.format_exc()
        message = str(sys.exc_info()[1])
        _write_flag(
            outdir,
            "FAILED",
            f"run_time_seconds={dt:.1f}\nhost={host}\n\n{message}\n\n{tb}",
        )
        (outdir / "RUNNING").unlink(missing_ok=True)
        print(f"[worker] FAILED after {dt:.1f}s -> {outdir}\n{tb}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))

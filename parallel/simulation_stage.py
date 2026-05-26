"""
Simulation-only subprocess for parallel BO workers.

This stage intentionally imports only the HOOMD simulation module, writes the
exact trajectory path returned by run_simulation(), flushes stdio, and exits
with os._exit(). The hard exit avoids fragile Python/HOOMD interpreter cleanup
from taking down the parent worker before downstream analysis can run.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path


_THIS_DIR = Path(__file__).resolve().parent
_PARENT = _THIS_DIR.parent
for p in (_PARENT, _THIS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from simulation import run_simulation  # noqa: E402


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def _jsonable_result(result: dict) -> dict:
    out = {}
    for key, value in result.items():
        try:
            json.dumps(value)
            out[key] = value
        except TypeError:
            out[key] = str(value)
    for key in ("gsd_path", "energy_csv"):
        if key in out and out[key]:
            out[key] = str(Path(out[key]).resolve())
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <config.json>", file=sys.stderr, flush=True)
        return 2

    cfg_path = Path(argv[1])
    cfg = json.loads(cfg_path.read_text())
    outdir = Path(cfg["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
        result = run_simulation(outdir=str(outdir), **cfg["run_kwargs"])
        payload = {
            "run_time_seconds": time.time() - t0,
            "host": os.uname().nodename,
            "result": _jsonable_result(result),
        }
        _write_json_atomic(outdir / "SIM_DONE.json", payload)
        print(
            f"[simulation_stage] wrote SIM_DONE.json "
            f"gsd={payload['result'].get('gsd_path', '<missing>')}",
            flush=True,
        )
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    except Exception:
        payload = {
            "run_time_seconds": time.time() - t0,
            "host": os.uname().nodename,
            "traceback": traceback.format_exc(),
        }
        _write_json_atomic(outdir / "SIM_FAILED.json", payload)
        print(payload["traceback"], file=sys.stderr, flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


if __name__ == "__main__":
    rc = main(sys.argv)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)

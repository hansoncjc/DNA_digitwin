"""
R6.9 r0 scan: one simulation per point of ``scan_config.POINTS``, scored
against the HS fluid target like an R6.6 evaluation.

    python run_point.py --list                      # points, table bounds, target md5; no simulation
    python run_point.py --index I --out SCAN_ROOT   # one point (Slurm array task)

Before anything is written, every point is checked: rmin < rmax < L/2
(``simulation.resolve_table_bounds``), n > m, and the target's md5; any
failure stops ``--list`` with exit code 1 and ``--index`` before the
simulation. Point I writes ``SCAN_ROOT/<id>/d0/`` laid out like a BO eval
folder (trajectory, ``S(q)_data/``, ``shift_rmse_diagnostics.json``,
``point_params.json``), so ``structure_metrics.py --gsd`` reads it as is.
``--steps`` and ``--device`` are for smoke tests only.
"""
import argparse
import hashlib
import inspect
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import scan_config as config  # noqa: E402
from simulation import resolve_table_bounds  # noqa: E402


def _md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def bounds(point):
    """Table bounds; raises ValueError for n <= m or rmax >= L/2."""
    if not point["n"] > point["m"]:
        raise ValueError(f"n={point['n']} must exceed m={point['m']}")
    b = resolve_table_bounds(point["U0"], point["n"], point["m"], point["r0"],
                             N=config.N, density=config.DENSITY)
    if not b["rmin"] < b["rmax"] < b["L"] / 2:
        raise ValueError(f"table bounds rmin={b['rmin']:.4g} rmax={b['rmax']:.4g} L/2={b['L'] / 2:.4g}")
    return {k: b[k] for k in ("rmin", "rmax", "t_tol", "L")}


def check_target():
    md5 = _md5(config.TARGET)
    if md5 != config.TARGET_MD5:
        raise ValueError(f"target md5 {md5} != {config.TARGET_MD5} ({config.TARGET})")
    return md5


def _git_head():
    """HEAD commit, with "-dirty" if tracked files differ from it."""
    try:
        head = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
        return head + ("-dirty" if dirty else "")
    except Exception:
        return None


def list_points():
    """Print every point with its table bounds; return the number that fail the checks."""
    failed = 0
    try:
        print(f"target {config.TARGET} md5 {check_target()}")
    except (OSError, ValueError) as e:
        failed += 1
        print(f"target FAILED: {e}")
    print(f"N {config.N}, density {config.DENSITY}, metric_kwargs {config.METRIC_KWARGS}")
    print(" #  id      U0      r0      n      m |   rmin   rmax  L/2   t_tol | role")
    for i, p in enumerate(config.POINTS):
        head = f"{i:2d} {p['id']:>3} {p['U0']:7.4f} {p['r0']:7.4f} {p['n']:6.2f} {p['m']:6.2f} |"
        try:
            b = bounds(p)
        except ValueError as e:
            failed += 1
            print(f"{head} FAILED: {e}")
            continue
        print(f"{head} {b['rmin']:6.4f} {b['rmax']:6.3f} {b['L'] / 2:4.0f} {b['t_tol']:.4g} | {p['role']}")
    print(f"{len(config.POINTS)} points, {failed} failed")
    return failed


def run_one(index, out_root, steps=None, device=None):
    from simulation import run_simulation
    from scattering import convert_to_SAXS_fft
    import numpy as np
    import metrics

    check_target()
    for p in config.POINTS:   # the whole set, so no array task runs if any point fails
        bounds(p)
    point = config.POINTS[index]
    sim_dir = Path(out_root) / point["id"] / "d0"
    sim_dir.mkdir(parents=True, exist_ok=True)

    run_kwargs = {"N": config.N, "plot": False}
    if steps is not None:
        run_kwargs["steps"] = int(steps)
    if device is not None:
        run_kwargs["device"] = device
    record = {
        "point": point,
        "table_bounds": bounds(point),
        "density": config.DENSITY,
        "run_kwargs": run_kwargs,
        "seed": inspect.signature(run_simulation).parameters["seed"].default,
        "target": str(config.TARGET),
        "target_md5": config.TARGET_MD5,
        "metric_kwargs": config.METRIC_KWARGS,
        "git_head": _git_head(),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    (sim_dir / "point_params.json").write_text(json.dumps(record, indent=2))

    result = run_simulation(
        density=config.DENSITY, U_0=point["U0"], r0=point["r0"], n=point["n"], m=point["m"],
        outdir=str(sim_dir), **run_kwargs,
    )
    convert_to_SAXS_fft(str(sim_dir))
    sim_sq = np.load(sim_dir / "S(q)_data" / "average_structure_factor.npy")
    target_sq = np.load(config.TARGET)
    # Writes shift_rmse_diagnostics.json and the overlay as a BO evaluation does;
    # raises metrics.MetricFailed (after writing the failed diagnostics) on a curve
    # without a usable peak.
    metrics._compare_shift_rmse(target_sq, sim_sq, str(sim_dir), None, config.METRIC_KWARGS)

    record["sim_result"] = {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v))
                            for k, v in dict(result).items()}
    (sim_dir / "point_params.json").write_text(json.dumps(record, indent=2))
    diag = json.loads((sim_dir / "shift_rmse_diagnostics.json").read_text())
    print(f"[r6.9] {point['id']} loss {diag['loss']:.6f} (M4 {diag['m4']:.6f}, "
          f"q1 ratio {diag['q1_ratio']}) -> {sim_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--index", type=int)
    ap.add_argument("--out", default="scan")
    ap.add_argument("--steps", type=int, default=None, help="smoke test only")
    ap.add_argument("--device", default=None, help="local test only (default: run_simulation's gpu)")
    a = ap.parse_args()
    if a.list:
        sys.exit(1 if list_points() else 0)
    if a.index is None:
        ap.error("--index or --list is required")
    run_one(a.index, a.out, a.steps, a.device)


if __name__ == "__main__":
    main()

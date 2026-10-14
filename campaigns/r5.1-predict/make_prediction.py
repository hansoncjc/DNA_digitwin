"""
R5.1 prediction: one simulation per condition with the R3.2 eval-56
coefficients (``predict_config.COEFFS``), forward model at function defaults
except ``N`` (seed: run_simulation's default, as in the R3.2 evaluations).

    python make_prediction.py --list                      # mapped parameters and checks, no simulation
    python make_prediction.py --index I --out PRED_ROOT   # one condition (Slurm array task)

Condition I is ``predict_config.CONDITIONS[I]`` (p0-p13). Before anything is
written, every condition is checked: ``check_mapped_params`` and the table
bounds of ``simulation.resolve_table_bounds`` (rmin < rmax, rmax < L/2); one
failing condition stops ``--list`` with exit code 1 and ``--index`` before
the simulation. Writes ``PRED_ROOT/<id>/`` (trajectory, S(q),
``pred_params.json``) and copies the S(q) to ``PRED_ROOT/<id>.npy``.
``--steps`` and ``--device`` are for smoke tests only.
"""
import argparse
import inspect
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import predict_config as config  # noqa: E402
from datasets import check_mapped_params, physics_map  # noqa: E402
from simulation import resolve_table_bounds  # noqa: E402


def mapped(cond):
    """Mapped (r0, U0, n, m) and table bounds; raises ValueError outside the checked range."""
    p = physics_map(cond["L_bridge"], cond["C_anchor"], cond["C_NaCl"], **config.COEFFS)
    check_mapped_params(p["U0"], p["n"], p["m"])
    b = resolve_table_bounds(p["U0"], p["n"], p["m"], p["r0"], N=config.N, density=config.DENSITY)
    return p, {k: b[k] for k in ("rmin", "rmax", "t_tol", "L")}


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


def list_conditions():
    """Print every condition with its mapped parameters; return the number that fail the checks."""
    print(f"coefficients {config.COEFFS}")
    print(f"  from {config.COEFFS_SOURCE}; density {config.DENSITY}, N {config.N}")
    print(" #  id  part  L_bridge C_anchor C_NaCl |     U0     r0      n      m |   rmin   rmax  L/2  t_tol   | range")
    failed = 0
    for i, c in enumerate(config.CONDITIONS):
        head = (f"{i:2d} {c['id']:>3} {c['part']} {c['L_bridge']:8.0f} {c['C_anchor']:6.0f} "
                f"{c['C_NaCl']:6.0f} |")
        try:
            p, b = mapped(c)
        except ValueError as e:
            failed += 1
            print(f"{head} FAILED: {e}")
            continue
        print(f"{head} {p['U0']:6.3f} {p['r0']:6.4f} {p['n']:6.3f} {p['m']:6.3f} | "
              f"{b['rmin']:6.4f} {b['rmax']:6.3f} {b['L'] / 2:4.0f} {b['t_tol']:.4g}  | {c['range']}")
    print(f"{len(config.CONDITIONS)} conditions, {failed} outside the checked range")
    return failed


def run_one(index, out_root, steps=None, device=None):
    from simulation import run_simulation
    from scattering import convert_to_SAXS_fft

    for c in config.CONDITIONS:   # the whole set, so no array task runs if any condition fails
        mapped(c)
    cond = config.CONDITIONS[index]
    p, bounds = mapped(cond)
    out_root = Path(out_root)
    sim_dir = out_root / cond["id"]
    sim_dir.mkdir(parents=True, exist_ok=True)

    run_kwargs = {"N": config.N, "plot": False}
    if steps is not None:
        run_kwargs["steps"] = int(steps)
    if device is not None:
        run_kwargs["device"] = device
    record = {
        "condition": cond,
        "coefficients": config.COEFFS,
        "coefficients_source": config.COEFFS_SOURCE,
        "mapped": p,
        "table_bounds": bounds,
        "density": config.DENSITY,
        "run_kwargs": run_kwargs,
        "seed": inspect.signature(run_simulation).parameters["seed"].default,
        "git_head": _git_head(),
    }
    (sim_dir / "pred_params.json").write_text(json.dumps(record, indent=2))

    result = run_simulation(
        density=config.DENSITY, U_0=p["U0"], r0=p["r0"], n=p["n"], m=p["m"],
        outdir=str(sim_dir), **run_kwargs,
    )
    convert_to_SAXS_fft(str(sim_dir))
    sq = sim_dir / "S(q)_data" / "average_structure_factor.npy"
    shutil.copyfile(sq, out_root / f"{cond['id']}.npy")

    record["sim_result"] = {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v))
                            for k, v in dict(result).items()}
    (sim_dir / "pred_params.json").write_text(json.dumps(record, indent=2))
    print(f"[r5.1] {cond['id']} done -> {out_root / (cond['id'] + '.npy')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--index", type=int)
    ap.add_argument("--out", default="predictions")
    ap.add_argument("--steps", type=int, default=None, help="smoke test only")
    ap.add_argument("--device", default=None, help="local test only (default: run_simulation's gpu)")
    a = ap.parse_args()
    if a.list:
        sys.exit(1 if list_conditions() else 0)
    if a.index is None:
        ap.error("--index or --list is required")
    run_one(a.index, a.out, a.steps, a.device)


if __name__ == "__main__":
    main()

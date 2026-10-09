"""
R3.1 ground truth: one simulation per condition with the ground-truth
coefficients (``gt_config.GROUND_TRUTH``), forward model at function defaults
except ``N`` and ``seed = gt_config.GT_SEED``.

    python make_ground_truth.py --list                      # mapped parameters, no simulation
    python make_ground_truth.py --index I --out GT_ROOT     # one condition (Slurm array task)

Condition I is ``gt_config.ALL_CONDITIONS[I]`` (0-23 training, 24-28 held-out).
Writes ``GT_ROOT/<id>/`` (trajectory, S(q), ``gt_params.json``) and copies the
S(q) to ``GT_ROOT/<id>.npy``, the file the BO datasets read (trim_tail=0).
``--steps`` and ``--device`` are for smoke tests only.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import gt_config as config  # noqa: E402
from datasets import check_mapped_params, physics_map  # noqa: E402


def mapped(cond, coeffs=None):
    coeffs = dict(config.GROUND_TRUTH if coeffs is None else coeffs)
    p = physics_map(cond["L_bridge"], cond["C_chol"], cond["C_NaCl"], **coeffs)
    check_mapped_params(p["U0"], p["n"], p["m"])
    return p


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
    print(f"ground truth {config.GROUND_TRUTH}, density {config.DENSITY}, N {config.N}")
    print(" #  id   role     L_bridge C_chol C_NaCl |     U0     r0      n      m")
    for i, c in enumerate(config.ALL_CONDITIONS):
        p = mapped(c)
        print(f"{i:2d} {c['id']:>3} {c['role']:<8} {c['L_bridge']:8.0f} {c['C_chol']:6.0f} "
              f"{c['C_NaCl']:6.0f} | {p['U0']:6.3f} {p['r0']:6.3f} {p['n']:6.2f} {p['m']:6.2f}")


def run_one(index, out_root, steps=None, device=None):
    from simulation import run_simulation
    from scattering import convert_to_SAXS_fft

    cond = config.ALL_CONDITIONS[index]
    p = mapped(cond)
    out_root = Path(out_root)
    sim_dir = out_root / cond["id"]
    sim_dir.mkdir(parents=True, exist_ok=True)

    run_kwargs = {"N": config.N, "seed": config.GT_SEED, "plot": False}
    if steps is not None:
        run_kwargs["steps"] = int(steps)
    if device is not None:
        run_kwargs["device"] = device
    record = {
        "condition": cond,
        "coefficients": config.GROUND_TRUTH,
        "mapped": p,
        "density": config.DENSITY,
        "run_kwargs": run_kwargs,
        "git_head": _git_head(),
    }
    (sim_dir / "gt_params.json").write_text(json.dumps(record, indent=2))

    result = run_simulation(
        density=config.DENSITY, U_0=p["U0"], r0=p["r0"], n=p["n"], m=p["m"],
        outdir=str(sim_dir), **run_kwargs,
    )
    convert_to_SAXS_fft(str(sim_dir))
    sq = sim_dir / "S(q)_data" / "average_structure_factor.npy"
    shutil.copyfile(sq, out_root / f"{cond['id']}.npy")

    record["sim_result"] = {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v))
                            for k, v in dict(result).items()}
    (sim_dir / "gt_params.json").write_text(json.dumps(record, indent=2))
    print(f"[gt] {cond['id']} done -> {out_root / (cond['id'] + '.npy')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--index", type=int)
    ap.add_argument("--out", default="ground_truth")
    ap.add_argument("--steps", type=int, default=None, help="smoke test only")
    ap.add_argument("--device", default=None, help="local test only (default: run_simulation's gpu)")
    a = ap.parse_args()
    if a.list:
        list_conditions()
        return
    if a.index is None:
        ap.error("--index or --list is required")
    run_one(a.index, a.out, a.steps, a.device)


if __name__ == "__main__":
    main()

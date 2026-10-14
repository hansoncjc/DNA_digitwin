"""
R3.2 best evaluation against the R3.1 ground truth: snapshots and S(q), one
file per training condition. Runs locally on downloaded copies.

    python plot_best_eval.py --gt GT_DIR --run RUN_DIR --eval 56
    python plot_best_eval.py --gt GT_DIR --run RUN_DIR --eval 56 --view perspective --ids d4 d16 --out DIR
    python plot_best_eval.py --gt GT_DIR --ids d4 --no-sq            # ground-truth snapshots only

GT_DIR is the R3.1 folder holding ``ground_truth/<id>/DNA_assembly_*.gsd`` and
``ground_truth/<id>.npy``; RUN_DIR is the R3.2 ``Optimization_Results`` copy
holding ``eval_XXX/<id>/``. Writes into OUT (default: this folder)::

    snapshots/gt/<id>.png          last frame of the ground truth, red
    snapshots/eval_XXX/<id>.png    last frame of the evaluation, green
    sq/<id>.png                    S(q), ground truth red, evaluation green

Snapshots come from the gsd-tools CLI (``gsd-tools render``) with its defaults
except the colour; ``--view`` overrides the camera preset when a figure must
match an earlier one. gsd-tools draws the box in the particle colour.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "r3.1-gt"))
from gt_config import TRAIN  # noqa: E402

GT_COLOR = "red"      # gsd-tools / Pillow name: #FF0000
EVAL_COLOR = "green"  # #008000
QA = r"$q\ (\mathring{\mathrm{A}}^{-1})$"


def gsd_tools_exe():
    exe = shutil.which("gsd-tools") or str(Path(sys.executable).parent / "gsd-tools")
    if not Path(exe).exists():
        raise FileNotFoundError("gsd-tools not found; run with the python of the digitwin env")
    return exe


def render(gsd_dir, out_png, color, view=None, size=None):
    gsd = sorted(glob.glob(str(Path(gsd_dir) / "DNA_assembly_*.gsd")))
    if not gsd:
        raise FileNotFoundError(f"no DNA_assembly_*.gsd in {gsd_dir}")
    cmd = [gsd_tools_exe(), "render", gsd[0], "-o", str(out_png), "--color", color, "--no-prompt"]
    if view is not None:
        cmd += ["--view", view]
    if size is not None:
        cmd += ["--size", str(size[0]), str(size[1])]
    out_png.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"{out_png}  <-  {gsd[0]}")


def condition_title(c):
    return (rf"{c['id']}: $L_{{\mathrm{{bridge}}}} = {c['L_bridge']:g}$ bp, "
            rf"$C_{{\mathrm{{anchor}}}} = {c['C_anchor']:g}$, $C_{{\mathrm{{NaCl}}}} = {c['C_NaCl']:g}$ mM")


def plot_sq(c, gt_npy, ev_npy, ev, out_png):
    g, s = np.load(gt_npy), np.load(ev_npy)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(g[:, 0], g[:, 1], color=GT_COLOR, alpha=0.8, ls="-", lw=1.8, label="GT")
    ax.loglog(s[:, 0], s[:, 1], color=EVAL_COLOR, alpha=0.8, ls="-", lw=1.8, label=f"trained (eval {ev})")
    ax.set_xlim(1.5e-3, 0.135)
    ax.set_ylim(0.07, 40)
    ax.set_xlabel(QA)
    ax.set_ylabel("S(q)")
    ax.set_title(condition_title(c))
    ax.grid(True, which="major", alpha=0.3, lw=0.5)
    ax.legend(loc="lower right", frameon=False)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="R3.1 folder (holds ground_truth/)")
    ap.add_argument("--run", default=None, help="R3.2 Optimization_Results copy")
    ap.add_argument("--eval", type=int, default=None, help="evaluation id, e.g. 56")
    ap.add_argument("--ids", nargs="+", default=None, help="condition ids (default: d0-d23)")
    ap.add_argument("--view", choices=["front", "perspective"], default=None,
                    help="gsd-tools camera preset (default: the gsd-tools default)")
    ap.add_argument("--size", type=int, nargs=2, default=None, metavar=("W", "H"),
                    help="snapshot size in pixels (default: the gsd-tools default)")
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--no-snapshots", action="store_true")
    ap.add_argument("--no-sq", action="store_true")
    a = ap.parse_args()
    if (a.run is None) != (a.eval is None):
        ap.error("--run and --eval go together")
    gt_root = Path(a.gt) / "ground_truth"
    out = Path(a.out)
    conds = [c for c in TRAIN if a.ids is None or c["id"] in a.ids]
    for c in conds:
        cid = c["id"]
        ev_dir = None if a.run is None else Path(a.run) / f"eval_{a.eval:03d}" / cid
        if not a.no_snapshots:
            render(gt_root / cid, out / "snapshots" / "gt" / f"{cid}.png", GT_COLOR, a.view, a.size)
            if ev_dir is not None:
                render(ev_dir, out / "snapshots" / f"eval_{a.eval:03d}" / f"{cid}.png", EVAL_COLOR, a.view, a.size)
        if not a.no_sq and ev_dir is not None:
            plot_sq(c, gt_root / f"{cid}.npy", ev_dir / "average_structure_factor.npy", a.eval,
                    out / "sq" / f"{cid}.png")


if __name__ == "__main__":
    main()

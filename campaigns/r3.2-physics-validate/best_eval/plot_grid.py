"""
R3.2 main figure: S(q) of one evaluation against the R3.1 ground truth on the
C_anchor x C_NaCl grid at one bridge length, with both snapshots in every panel.

    python plot_grid.py --gt GT_DIR --run RUN_DIR --eval 56 --L 80

Rows: C_anchor (anchoring DNA coverage) rising from bottom to top; columns: C_NaCl rising from left to
right. Each panel (aspect 8:6): S(q), ground truth red, evaluation green,
alpha 0.8, log-log, legend lower right; the two snapshots in the upper right
(left: ground truth, right: trained). Snapshots are read from
snapshots/gt/<id>.png and snapshots/eval_XXX/<id>.png made by
plot_best_eval.py (--size 1000 1000). Writes grid_L<L>_eval_XXX.png here.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "r3.1-gt"))
from gt_config import TRAIN  # noqa: E402

GT_COLOR, EVAL_COLOR = "red", "green"
QA = r"$q\ (\mathring{\mathrm{A}}^{-1})$"
INSETS = [0.585, 0.79]          # x0 of the two insets (axes fraction)
INSET_Y, INSET_W, INSET_H = 0.69, 0.20, 0.30


def panel(ax, c, gt_root, ev_dir, snap_dir, ev):
    cid = c["id"]
    g = np.load(gt_root / f"{cid}.npy")
    s = np.load(ev_dir / cid / "average_structure_factor.npy")
    ax.loglog(g[:, 0], g[:, 1], color=GT_COLOR, alpha=0.8, lw=1.6, label="Ground Truth")
    ax.loglog(s[:, 0], s[:, 1], color=EVAL_COLOR, alpha=0.8, lw=1.6, label=f"Trained (eval {ev})")
    ax.set_xlim(1.5e-3, 0.135)
    ax.set_ylim(0.07, 40)
    ax.grid(True, which="major", alpha=0.3, lw=0.5)
    ax.set_title(rf"{cid}: $L_{{\mathrm{{bridge}}}} = {c['L_bridge']:g}$ bp, "
                 rf"$C_{{\mathrm{{anchor}}}} = {c['C_anchor']:g}$, $C_{{\mathrm{{NaCl}}}} = {c['C_NaCl']:g}$ mM",
                 fontsize=10)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    for x0, (sub, lab) in zip(INSETS, (("gt", "Ground Truth"), (f"eval_{ev:03d}", "Trained"))):
        ins = ax.inset_axes([x0, INSET_Y, INSET_W, INSET_H])
        ins.imshow(plt.imread(snap_dir / sub / f"{cid}.png"))
        ins.set_xticks([]); ins.set_yticks([])
        for sp in ins.spines.values():
            sp.set_visible(False)
        ins.text(0.5, -0.03, lab, transform=ins.transAxes, ha="center", va="top", fontsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--eval", type=int, required=True)
    ap.add_argument("--L", type=float, required=True, help="bridge length (bp)")
    ap.add_argument("--out", default=str(HERE))
    a = ap.parse_args()
    conds = [c for c in TRAIN if c["L_bridge"] == a.L]
    anchor = sorted({c["C_anchor"] for c in conds})[::-1]     # top row = highest C_anchor
    salt = sorted({c["C_NaCl"] for c in conds})
    fig, axes = plt.subplots(len(anchor), len(salt), figsize=(4 * len(salt), 3 * len(anchor)),
                             sharex=True, sharey=True)
    ev_dir = Path(a.run) / f"eval_{a.eval:03d}"
    for i, an in enumerate(anchor):
        for j, sa in enumerate(salt):
            c = next(c for c in conds if c["C_anchor"] == an and c["C_NaCl"] == sa)
            panel(axes[i, j], c, Path(a.gt) / "ground_truth", ev_dir, HERE / "snapshots", a.eval)
            if j == 0:
                axes[i, j].set_ylabel("S(q)")
            if i == len(anchor) - 1:
                axes[i, j].set_xlabel(QA)
    fig.tight_layout(rect=(0.035, 0.045, 1, 1))
    fig.supxlabel(r"Salt Concentration, $C_{\mathrm{NaCl}}$ (mM)", fontsize=14, y=0.0)
    fig.supylabel(r"Anchoring DNA Coverage, $C_{\mathrm{anchor}}$ (molecules / particle)", fontsize=14, x=0.0)
    # direction of increase along the big axes: salt to the right, coverage upward
    arrow = dict(arrowstyle="-|>,head_length=1.2,head_width=0.6", lw=4.5, color=EVAL_COLOR,
                 mutation_scale=12, transform=fig.transFigure)
    fig.add_artist(FancyArrowPatch((0.06, 0.04), (0.99, 0.04), **arrow))
    fig.add_artist(FancyArrowPatch((0.03, 0.06), (0.03, 0.99), **arrow))
    out = Path(a.out) / f"grid_L{a.L:g}_eval_{a.eval:03d}.png"
    fig.savefig(out, dpi=200)
    print(out)


if __name__ == "__main__":
    main()

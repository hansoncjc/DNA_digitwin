"""
R5.1 grid figures: predicted S(q) on the condition grid, in the layout of the
R3.2 main figure (``../../r3.2-physics-validate/best_eval/plot_grid.py``).

    python plot_grid_r51.py --part part1 --pred PRED_RUN --gt GT_DIR --run R32_RUN

part1: L_bridge = 80, rows C_anchor (rising upward), columns C_NaCl (rising to
the right). part2: C_anchor = 100, rows L_bridge, columns C_NaCl.

A training cell (d0-d23) shows the R3.1 ground truth (red) and the R3.2
eval-56 S(q) (green), with both snapshots, as in the R3.2 figure. Every other
cell is a prediction (p0-p15) and shows the predicted S(q) only (blue) and its
snapshot. The R3.1 held-out curves (h0, h3 lie on the part-2 grid) are not
drawn: those cells show the predictions p14, p15 (author, 2026-10-17).
Each panel: log-log, alpha 0.8 where two curves overlap and 1 for a single
curve, legend lower right, snapshots upper right
(gsd-tools ``render``, front view, defaults except the colour, which matches
the line). Big arrows: increasing condition along each axis; same width and
margins as the R3.2 figure, length set by the grid.

PRED_RUN holds predictions/<id>/ (R5.1 download); GT_DIR is the R3.1 folder
holding ground_truth/; R32_RUN is the R3.2 Optimization_Results copy holding
eval_056/. All read only. Snapshots are rendered once into snapshots/{gt,
eval_056,pred}/<id>.png here (--rerender to redo); the figure is written here
as grid_<part>.png.
"""
import argparse
import glob
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import predict_config as config  # noqa: E402
from predict_config import GT_CONDITIONS  # noqa: E402

EVAL = 56
GT_COLOR, EVAL_COLOR, PRED_COLOR = "red", "green", "#1F77B4"   # gsd-tools / Pillow colours
QA = r"$q\ (\mathring{\mathrm{A}}^{-1})$"
INSETS = [0.585, 0.79]          # x0 of the two insets (axes fraction), as in R3.2
INSET_Y, INSET_W, INSET_H = 0.69, 0.20, 0.30
PANEL_W, PANEL_H = 4.0, 3.0     # inches per panel, as in R3.2

# Margins of the R3.2 figure (16 x 9 in), in inches, so every grid keeps them.
M_LEFT, M_BOTTOM = 0.035 * 16, 0.045 * 9
ARROW_H_Y, ARROW_H_X0, ARROW_H_RIGHT = 0.04 * 9, 0.06 * 16, 0.01 * 16
ARROW_V_X, ARROW_V_Y0, ARROW_V_TOP = 0.03 * 16, 0.06 * 9, 0.01 * 9

AXES = {
    "L_bridge": (r"Bridge Length, $L_{\mathrm{bridge}}$ (bp)"),
    "C_anchor": (r"Anchoring DNA Coverage, $C_{\mathrm{anchor}}$ (molecules / particle)"),
    "C_NaCl": (r"Salt Concentration, $C_{\mathrm{NaCl}}$ (mM)"),
}
PARTS = {   # fixed condition, row variable, column variable
    "part1": ({"L_bridge": config.PART1_L_BRIDGE}, "C_anchor", "C_NaCl"),
    "part2": ({"C_anchor": config.PART2_C_ANCHOR}, "L_bridge", "C_NaCl"),
}


def gsd_tools_exe():
    exe = shutil.which("gsd-tools") or str(Path(sys.executable).parent / "gsd-tools")
    if not Path(exe).exists():
        raise FileNotFoundError("gsd-tools not found; run with the python of the digitwin env")
    return exe


def render(gsd_dir, out_png, color, rerender=False):
    if out_png.exists() and not rerender:
        return
    gsd = sorted(glob.glob(str(Path(gsd_dir) / "DNA_assembly_*.gsd")))
    if not gsd:
        raise FileNotFoundError(f"no DNA_assembly_*.gsd in {gsd_dir}")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cmd = [gsd_tools_exe(), "render", gsd[0], "-o", str(out_png), "--color", color, "--no-prompt"]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"{out_png}  <-  {gsd[0]}")


def condition_title(c):
    return (rf"{c['id']}: $L_{{\mathrm{{bridge}}}} = {c['L_bridge']:g}$ bp, "
            rf"$C_{{\mathrm{{anchor}}}} = {c['C_anchor']:g}$, $C_{{\mathrm{{NaCl}}}} = {c['C_NaCl']:g}$ mM")


def cell_curves(c, gt_root, r32_eval, pred_root, snap_dir, rerender):
    """[(S(q) array, colour, legend label, snapshot png, inset label)] for one cell."""
    cid = c["id"]
    if cid.startswith("p"):
        render(pred_root / cid, snap_dir / "pred" / f"{cid}.png", PRED_COLOR, rerender)
        return [(np.load(pred_root / f"{cid}.npy"), PRED_COLOR, f"Predicted (eval {EVAL})",
                 snap_dir / "pred" / f"{cid}.png", "Predicted")]
    render(gt_root / cid, snap_dir / "gt" / f"{cid}.png", GT_COLOR, rerender)
    out = [(np.load(gt_root / f"{cid}.npy"), GT_COLOR, "Ground Truth",
            snap_dir / "gt" / f"{cid}.png", "Ground Truth")]
    ev_dir = r32_eval / cid
    if ev_dir.exists():
        render(ev_dir, snap_dir / f"eval_{EVAL:03d}" / f"{cid}.png", EVAL_COLOR, rerender)
        out.append((np.load(ev_dir / "average_structure_factor.npy"), EVAL_COLOR,
                    f"Trained (eval {EVAL})", snap_dir / f"eval_{EVAL:03d}" / f"{cid}.png", "Trained"))
    return out


def panel(ax, c, curves):
    alpha = 0.8 if len(curves) > 1 else 1.0     # translucent only where curves overlap
    for s, color, label, _, _ in curves:
        ax.loglog(s[:, 0], s[:, 1], color=color, alpha=alpha, lw=1.6, label=label)
    ax.set_xlim(1.5e-3, 0.135)
    ax.set_ylim(0.07, 40)
    ax.grid(True, which="major", alpha=0.3, lw=0.5)
    ax.set_title(condition_title(c), fontsize=10)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    # snapshots fill the insets from the right
    for x0, (_, _, _, png, lab) in zip(INSETS[len(INSETS) - len(curves):], curves):
        ins = ax.inset_axes([x0, INSET_Y, INSET_W, INSET_H])
        ins.imshow(plt.imread(png))
        ins.set_xticks([]); ins.set_yticks([])
        for sp in ins.spines.values():
            sp.set_visible(False)
        ins.text(0.5, -0.03, lab, transform=ins.transAxes, ha="center", va="top", fontsize=7)


def grid_conditions(part):
    fixed, row_var, col_var = PARTS[part]
    train = [c for c in GT_CONDITIONS if c["role"] == "train"]
    pool = [c for c in train + config.CONDITIONS
            if all(c[k] == v for k, v in fixed.items())]
    if part == "part1":
        rows, cols = config.PART1_C_ANCHOR, config.PART1_C_NACL
    else:
        rows, cols = config.PART2_L_BRIDGE, config.PART2_C_NACL
    grid = {}
    for r in rows:
        for col in cols:
            grid[r, col] = next(c for c in pool if c[row_var] == r and c[col_var] == col)
    return sorted(rows)[::-1], sorted(cols), grid, row_var, col_var


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=sorted(PARTS), required=True)
    ap.add_argument("--pred", required=True, help="R5.1 download (holds predictions/)")
    ap.add_argument("--gt", required=True, help="R3.1 folder (holds ground_truth/)")
    ap.add_argument("--run", required=True, help="R3.2 Optimization_Results copy (holds eval_056/)")
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--rerender", action="store_true")
    a = ap.parse_args()

    rows, cols, grid, row_var, col_var = grid_conditions(a.part)
    nr, nc = len(rows), len(cols)
    fw, fh = PANEL_W * nc, PANEL_H * nr
    fig, axes = plt.subplots(nr, nc, figsize=(fw, fh), sharex=True, sharey=True)
    for i, r in enumerate(rows):            # top row = largest value
        for j, col in enumerate(cols):
            c = grid[r, col]
            curves = cell_curves(c, Path(a.gt) / "ground_truth", Path(a.run) / f"eval_{EVAL:03d}",
                                 Path(a.pred) / "predictions", HERE / "snapshots", a.rerender)
            panel(axes[i, j], c, curves)
            if j == 0:
                axes[i, j].set_ylabel("S(q)")
            if i == nr - 1:
                axes[i, j].set_xlabel(QA)
    fig.tight_layout(rect=(M_LEFT / fw, M_BOTTOM / fh, 1, 1))
    fig.supxlabel(AXES[col_var], fontsize=14, y=0.0)
    fig.supylabel(AXES[row_var], fontsize=14, x=0.0)
    # direction of increase along the big axes: columns to the right, rows upward
    arrow = dict(arrowstyle="-|>,head_length=1.2,head_width=0.6", lw=4.5, color=PRED_COLOR,
                 mutation_scale=12, transform=fig.transFigure)
    fig.add_artist(FancyArrowPatch((ARROW_H_X0 / fw, ARROW_H_Y / fh),
                                   (1 - ARROW_H_RIGHT / fw, ARROW_H_Y / fh), **arrow))
    fig.add_artist(FancyArrowPatch((ARROW_V_X / fw, ARROW_V_Y0 / fh),
                                   (ARROW_V_X / fw, 1 - ARROW_V_TOP / fh), **arrow))
    out = Path(a.out) / f"grid_{a.part}.png"
    fig.savefig(out, dpi=200)
    print(out)


if __name__ == "__main__":
    main()

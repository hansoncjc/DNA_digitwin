"""
Figures for the R3.2 note (vault 02_Logs/02_testings/2026-10-09 BO_new_mapping_all_coeff), Results 2-4.

Usage: python plot_r32.py RUN_DIR GT_DIR OUT_DIR [--floor 0.599] [--floor-csv CSV] [--threshold T]

RUN_DIR: downloaded Optimization_Results copy; GT_DIR: Sweep8_R3.1_GT (both
read only). r32_evals.csv and r32_conditions.csv are read from this folder
(summarize_r32.py). --floor draws the truth-loss floor (total over 24
conditions), --threshold the success threshold once it is set. Writes
r32_loss_history.png, r32_per_condition.png and r32_sq_best_d0_d23.png into
OUT_DIR. The S(q) main figure (3 x 4 grid with snapshots) is
../best_eval/plot_grid.py.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

BLUE, ORANGE, AQUA, YELLOW, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a5cd6", "#d23c3c"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e6e5e1", "#fcfcfb"
DARK = {20.0: "#123f7a", 80.0: "#8f3311"}
L_COLOR = {20.0: BLUE, 80.0: ORANGE}
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.6,
    "savefig.dpi": 160,
})
QA = r"$q\ (\mathring{\mathrm{A}}^{-1})$"
COEFFS = ["k", "A", "K_s", "a_m", "delta", "a_n"]
GT = {"k": 0.65, "A": 1.2, "K_s": 0.07, "a_m": 3.2, "delta": 3.2, "a_n": 6.4}
BOX = {"k": (0.40, 0.90), "A": (0.85, 2.50), "K_s": (0.0, 0.10), "a_m": (2.91, 4.00),
       "delta": (1.09, 4.00), "a_n": (4.00, 8.00)}


def gt_sq(gt_dir, cid):
    return np.load(Path(gt_dir) / "ground_truth" / f"{cid}.npy")


def run_sq(run, ev, cid):
    return np.load(Path(run) / f"eval_{ev:03d}" / cid / "average_structure_factor.npy")


def fig_loss_history(ev, out, floor, threshold):
    best = int(ev.loss.idxmin())
    rep = ev[ev.repeat_set]
    fig = plt.figure(figsize=(13, 7.6))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.35, 1], hspace=0.42, wspace=0.42)
    ax = fig.add_subplot(gs[0, :])
    d2 = ev.d2_sim_dispersed.astype(bool)
    ax.plot(ev.index, ev.loss, color=MUTED, lw=0.8, zorder=1)
    ax.scatter(ev.index[~d2], ev.loss[~d2], s=16, color=INK2, zorder=2, label="total loss (24 conditions)")
    ax.scatter(ev.index[d2], ev.loss[d2], s=22, facecolor="none", edgecolor=ORANGE, lw=1.1, zorder=2,
               label="d2 simulated dispersed (target peaked)")
    ax.scatter(rep.index, rep.loss, s=30, facecolor="none", edgecolor=BLUE, lw=1.1, zorder=3,
               label=f"repeat set ({len(rep)} evaluations at nearly one point)")
    ax.step(ev.index, ev.best_so_far, where="post", color=RED, lw=1.6, label="best so far", zorder=4)
    ax.axhspan(rep.loss.mean() - rep.loss.std(ddof=1), rep.loss.mean() + rep.loss.std(ddof=1),
               color=BLUE, alpha=0.10, lw=0)
    ax.axhline(rep.loss.mean(), color=BLUE, lw=0.8, ls=":")
    ax.text(31, rep.loss.mean() * 0.955, f"repeat set {rep.loss.mean():.3f} +- {rep.loss.std(ddof=1):.3f} (band: +- 1 sd)",
            color=BLUE, fontsize=8, va="top")
    if floor is not None:
        ax.axhline(floor, color=INK, lw=1.0, ls="--")
        ax.text(1, floor * 0.97, f"truth-loss floor {floor:.3f} (one truth evaluation)", fontsize=8, va="top")
    if threshold is not None:
        ax.axhline(threshold, color=AQUA, lw=1.2)
        ax.text(ev.index.max(), threshold * 1.03, f"success threshold {threshold:g}", color=AQUA,
                fontsize=8, ha="right", va="bottom")
    ax.annotate(f"eval {best}: {ev.loss[best]:.3f}", (best, ev.loss[best]), xytext=(best - 14, 0.45),
                fontsize=8, color=RED, arrowprops=dict(arrowstyle="-", color=RED, lw=0.7))
    ax.set_yscale("log")
    ax.set_ylim(0.4, 8)
    ax.set_yticks([0.5, 0.75, 1, 2, 4, 8], ["0.5", "0.75", "1", "2", "4", "8"])
    ax.minorticks_off()
    ax.set_xlim(-1, ev.index.max() + 1)
    ax.set_xlabel("evaluation")
    ax.set_ylabel("total shift_rmse loss")
    ax.set_title("(a) loss history (eval 0 = initial point; 80 of 100 evaluations before the walltime)",
                 loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
    for i, c in enumerate(COEFFS):
        a = fig.add_subplot(gs[1, i])
        a.axhline(GT[c], color=INK, lw=1.0, ls="--")
        a.plot(ev.index, ev[c], color=MUTED, lw=0.7)
        a.scatter(ev.index, ev[c], s=6, color=INK2, zorder=2)
        a.scatter([best], [ev.loc[best, c]], s=30, color=RED, zorder=3)
        a.set_ylim(*BOX[c])
        a.set_xlabel("evaluation")
        rel = 100 * (ev.loc[best, c] - GT[c]) / GT[c]
        a.set_title(f"{c}: best {ev.loc[best, c]:.4g} ({rel:+.1f} %)", loc="left", fontsize=8.5)
    fig.text(0.5, 0.005, "(b) coefficients; y range = search box (a_n = a_m + delta: implied range); "
             "dashed = ground truth; red = best evaluation", ha="center", fontsize=8.5, color=INK2)
    fig.savefig(out / "r32_loss_history.png", bbox_inches="tight")
    plt.close(fig)


def fig_per_condition(cond, out, floor_csv):
    anchor = sorted(cond.C_anchor.unique())
    salt = sorted(cond.C_NaCl.unique())
    norm = LogNorm(1e-3, 0.15)
    cmap = plt.get_cmap("Blues")
    fl = pd.read_csv(floor_csv).set_index("id").loss if floor_csv else None
    fig = plt.figure(figsize=(12.5, 8.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.85], hspace=0.38)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    for ax, L in zip(axes, (20.0, 80.0)):
        sub = cond[cond.L_bridge == L]
        grid = np.zeros((len(anchor), len(salt)))
        for _, r in sub.iterrows():
            grid[anchor.index(r.C_anchor), salt.index(r.C_NaCl)] = r.loss_best
        im = ax.imshow(grid, origin="lower", cmap=cmap, norm=norm, aspect="auto")
        for _, r in sub.iterrows():
            i, j = anchor.index(r.C_anchor), salt.index(r.C_NaCl)
            tc = "white" if r.loss_best > 0.03 else INK
            ax.text(j, i + 0.28, r.id, ha="center", va="center", fontsize=8.5, color=tc, fontweight="bold")
            ax.text(j, i + 0.06, f"best {r.loss_best:.3f}", ha="center", va="center", fontsize=8, color=tc)
            ax.text(j, i - 0.14, f"rep {r.rep_mean:.3f} +- {r.rep_sd:.3f}", ha="center", va="center",
                    fontsize=7.5, color=tc)
            ax.text(j, i - 0.33, f"U0 {r.U0_best:.2f} / {r.U0_gt:.2f}", ha="center", va="center",
                    fontsize=7.5, color=tc)
        ax.set_xticks(range(len(salt)), [f"{s:g}" for s in salt])
        ax.set_yticks(range(len(anchor)), [f"{c:g}" for c in anchor])
        ax.set_xlabel("C_NaCl (mM)"); ax.set_ylabel(r"$C_{\mathrm{anchor}}$ (molecules / particle)")
        ax.grid(False)
        r0 = sub.iloc[0]
        ax.set_title(f"({'a' if L == 20.0 else 'b'}) L_bridge = {L:g} bp: r0 {r0.r0_best:.3f} / {r0.r0_gt:.3f}, n {r0.n_best:.2f} / {r0.n_gt:.2f}, "
                     f"m {r0.m_best:.2f} / {r0.m_gt:.2f} (best / GT)", loc="left", fontsize=8.5)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02).set_label("loss at the best evaluation")
    ax = fig.add_subplot(gs[1, :])
    cond = cond.copy()
    cond["i"] = cond.id.str[1:].astype(int)
    cond = cond.sort_values("i")
    x = cond.i.values
    ax.vlines(x, cond.rep_min, cond.rep_max, color=BLUE, lw=1.0, alpha=0.6)
    ax.errorbar(x, cond.rep_mean, yerr=cond.rep_sd, fmt="o", ms=4, color=BLUE, lw=1.4, capsize=2,
                label="repeat set: mean +- sd (line: min to max)")
    ax.scatter(x + 0.18, cond.loss_best, s=22, color=RED, zorder=3, label="best evaluation")
    if fl is not None:
        ax.scatter(x - 0.18, fl.reindex(cond.id).values, s=22, marker="D", color=INK, zorder=3,
                   label="truth-loss floor")
    ax.set_yscale("log")
    ax.set_xticks(x, cond.id, fontsize=8)
    ax.set_ylabel("per-condition loss")
    for i in x:
        if cond.set_index("i").loc[i, "L_bridge"] == 80.0:
            ax.axvspan(i - 0.5, i + 0.5, color=ORANGE, alpha=0.05, lw=0)
    ax.set_xlim(-0.6, 23.6)
    ax.set_title("(c) per-condition loss; shaded = L_bridge 80; the six gas-like conditions sit near 0.002",
                 loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.text(0.01, 0.955, "(a, b) cell text: id / loss at the best evaluation / repeat-set mean +- sd / "
             "U0 best / GT", fontsize=8.5, color=INK2)
    fig.savefig(out / "r32_per_condition.png", bbox_inches="tight")
    plt.close(fig)


def _sq_axes(ax):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1.5e-3, 0.135); ax.set_ylim(0.07, 40)
    ax.axhline(1, color=MUTED, lw=0.6)


def fig_sq_all(cond, run, gt_dir, best, out):
    c = cond.copy()
    c["i"] = c.id.str[1:].astype(int)
    c = c.sort_values("i")
    fig, axes = plt.subplots(4, 6, figsize=(16, 10.5), sharex=True, sharey=True)
    for ax, (_, r) in zip(axes.flat, c.iterrows()):
        g = gt_sq(gt_dir, r.id)
        b = run_sq(run, best, r.id)
        ax.plot(g[:, 0], g[:, 1], color=L_COLOR[r.L_bridge], lw=3.0, alpha=0.45)
        ax.plot(b[:, 0], b[:, 1], color=DARK[r.L_bridge], lw=1.0)
        _sq_axes(ax)
        ax.set_title(f"{r.id}: L={r.L_bridge:g}, C_anchor={r.C_anchor:g}, NaCl={r.C_NaCl:g}\n"
                     f"U0 {r.U0_best:.2f} / {r.U0_gt:.2f}; loss {r.loss_best:.3f}", fontsize=8, loc="left")
    for ax in axes[-1, :]:
        ax.set_xlabel(QA)
    for ax in axes[:, 0]:
        ax.set_ylabel("S(q)")
    handles = [Line2D([], [], color=MUTED, lw=3, alpha=0.6), Line2D([], [], color=INK, lw=1)]
    fig.legend(handles, ["ground truth (seed 1)", f"best evaluation, eval {best} (seed 42)"],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.savefig(out / "r32_sq_best_d0_d23.png", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("gt"); ap.add_argument("out")
    ap.add_argument("--floor", type=float, default=None)
    ap.add_argument("--floor-csv", default=None, help="per-condition floor: columns id, loss")
    ap.add_argument("--threshold", type=float, default=None)
    a = ap.parse_args()
    out = Path(a.out)
    here = Path(__file__).resolve().parent
    ev = pd.read_csv(here / "r32_evals.csv", index_col="eval")
    cond = pd.read_csv(here / "r32_conditions.csv")
    best = int(ev.loss.idxmin())
    fig_loss_history(ev, out, a.floor, a.threshold)
    fig_per_condition(cond, out, a.floor_csv)
    fig_sq_all(cond, a.run, a.gt, best, out)
    print("figures written to", out)


if __name__ == "__main__":
    main()

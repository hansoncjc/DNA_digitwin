"""
Figures for the R3.1 ground-truth note (2026-10-09 Sim_Mie_r0_Sweep01).

Usage: python plot_r31.py RUN_DIR SUMMARY_CSV OUT_DIR

RUN_DIR is the downloaded Sweep8_R3.1_GT folder (read only); SUMMARY_CSV is
the output of summarize_r31.py. Writes PNG files into OUT_DIR.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e6e5e1", "#fcfcfb"
L_COLOR = {20.0: BLUE, 80.0: ORANGE}
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.6,
    "savefig.dpi": 160,
})
QA = r"$q\ (\mathring{\mathrm{A}}^{-1})$"
DELTA = 0.5          # shift_rmse dispersed_delta
SEARCH = (0.006, 0.12)
SQ_WINDOW_STEPS = 100 * 50_000


def load_sq(run, cid):
    return np.load(Path(run) / "ground_truth" / f"{cid}.npy")


def fig_condition_map(df, out):
    tr = df[df.role == "train"]
    anchor = sorted(tr.C_anchor.unique())
    salt = sorted(tr.C_NaCl.unique())
    norm = Normalize(0, 7)
    cmap = plt.get_cmap("Blues")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for ax, L in zip(axes, (20.0, 80.0)):
        sub = tr[tr.L_bridge == L]
        r0, n, m = sub.r0.iloc[0], sub.n.iloc[0], sub.m.iloc[0]
        grid = np.zeros((len(anchor), len(salt)))
        for _, r in sub.iterrows():
            grid[anchor.index(r.C_anchor), salt.index(r.C_NaCl)] = r.U0
        ax.imshow(grid, origin="lower", cmap=cmap, norm=norm, aspect="auto")
        for _, r in sub.iterrows():
            i, j = anchor.index(r.C_anchor), salt.index(r.C_NaCl)
            near = "NEAR" in r.qc_status
            if r.dispersed:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="///",
                                           edgecolor=MUTED, lw=0))
            txt_c = "white" if r.U0 > 4 else INK
            ax.text(j, i + 0.25, r.id, ha="center", va="center", fontsize=8.5, color=txt_c, fontweight="bold")
            ax.text(j, i + 0.0, f"U0 = {r.U0:.2f}", ha="center", va="center", fontsize=8, color=txt_c)
            ax.text(j, i - 0.25, f"{r.s_max_minus_1:.2f}" + (" NEAR" if near else ""),
                    ha="center", va="center", fontsize=8,
                    color=txt_c, fontweight="bold" if near else "normal")
        ax.set_xticks(range(len(salt)), [f"{s:g}" for s in salt])
        ax.set_yticks(range(len(anchor)), [f"{c:g}" for c in anchor])
        ax.set_xlabel("C_NaCl (mM)")
        ax.set_ylabel(r"$C_{\mathrm{anchor}}$ (molecules / particle)")
        ax.grid(False)
        ax.set_title(f"L_bridge = {L:g} bp:  r0 = {r0:.3f},  n = {n:.2f},  m = {m:.2f}",
                     loc="left", fontsize=9.5)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("U0 (kT)")
    fig.text(0.01, -0.06, "Cell text: id / U0 / smoothed s_max - 1 in the first-peak search window. "
             "Hatched: dispersed (s_max - 1 < 0.5). NEAR: |s_max - 1 - 0.5| < 0.25.",
             fontsize=8, color=INK2)
    fig.savefig(out / "r31_condition_map.png", bbox_inches="tight")
    plt.close(fig)


def fig_sq_grid(df, run, out):
    tr = df[df.role == "train"]
    anchor = sorted(tr.C_anchor.unique())[::-1]       # high loading on top
    salt = sorted(tr.C_NaCl.unique())
    fig, axes = plt.subplots(len(anchor), len(salt), figsize=(12, 8), sharex=True, sharey=True)
    for i, c in enumerate(anchor):
        for j, s in enumerate(salt):
            ax = axes[i, j]
            for L in (20.0, 80.0):
                r = tr[(tr.C_anchor == c) & (tr.C_NaCl == s) & (tr.L_bridge == L)].iloc[0]
                sq = load_sq(run, r.id)
                ax.plot(sq[:, 0], sq[:, 1], color=L_COLOR[L], lw=1.3,
                        ls="--" if r.dispersed else "-")
                ax.text(0.97, 0.93 - (0.11 if L == 80.0 else 0), f"{r.id}: U0={r.U0:.2f}",
                        transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=L_COLOR[L])
            ax.axvspan(*SEARCH, color=GRID, alpha=0.0)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlim(1.5e-3, 0.135); ax.set_ylim(0.07, 40)
            ax.axhline(1, color=MUTED, lw=0.6)
            if i == 0:
                ax.set_title(f"C_NaCl = {s:g} mM", fontsize=9.5)
            if j == 0:
                ax.set_ylabel(f"$C_{{\\mathrm{{anchor}}}}$ = {c:g}\nS(q)")
            if i == len(anchor) - 1:
                ax.set_xlabel(QA)
    handles = [Line2D([], [], color=BLUE), Line2D([], [], color=ORANGE),
               Line2D([], [], color=INK2, ls="--")]
    fig.legend(handles, ["L_bridge = 20 bp", "L_bridge = 80 bp", "dashed: dispersed by the QC rule"],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "r31_sq_grid.png", bbox_inches="tight")
    plt.close(fig)


def fig_transition(df, out):
    tr = df[df.role == "train"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    ax = axes[0]
    for L in (20.0, 80.0):
        sub = tr[tr.L_bridge == L].sort_values("U0")
        ax.plot(sub.U0, sub.s_max_minus_1, color=L_COLOR[L], marker="o", ms=5,
                label=f"L_bridge = {L:g} bp (m = {sub.m.iloc[0]:.2f})")
        for _, r in sub.iterrows():
            if "NEAR" in r.qc_status:
                ax.annotate(r.id, (r.U0, r.s_max_minus_1), textcoords="offset points",
                            xytext=(6, -10), fontsize=8, color=L_COLOR[L])
    ho = df[df.role == "heldout"]
    ax.scatter(ho.U0, ho.s_max_minus_1, marker="x", color=INK2, zorder=3, label="held-out")
    ax.axhline(DELTA, color=MUTED, ls="--", lw=1)
    ax.text(6.4, DELTA + 0.06, "dispersed threshold 0.5", ha="right", fontsize=8, color=MUTED)
    ax.set_xlabel("U0 (kT)"); ax.set_ylabel("smoothed s_max - 1 in search window")
    ax.set_title("(a) first-peak height against U0", loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    ax = axes[1]
    agg = df[~df.dispersed]
    for L, col in ((20.0, BLUE), (40.0, AQUA), (80.0, ORANGE), (120.0, YELLOW)):
        sub = agg[agg.L_bridge == L]
        if len(sub):
            mk = "o" if sub.role.iloc[0] == "train" and L in (20.0, 80.0) else "x"
            ax.scatter(sub.U0, sub.q1_times_r0, color=col, marker=mk, s=28,
                       label=f"L_bridge = {L:g} (r0 = {sub.r0.iloc[0]:.2f})")
    ax.set_xlabel("U0 (kT)"); ax.set_ylabel(r"$q_1 r_0$ ($\mathring{\mathrm{A}}^{-1}\,\sigma$)")
    ax.set_title("(b) first-peak position times r0 (non-dispersed only)", loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "r31_transition_q1.png", bbox_inches="tight")
    plt.close(fig)


def fig_energy(df, run, out):
    norm = Normalize(df.U0.min(), df.U0.max())
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), gridspec_kw={"width_ratios": [2.2, 1]})
    for _, r in df.iterrows():
        e = np.genfromtxt(Path(run) / "ground_truth" / r.id / "potential_energy.csv",
                          names=True, delimiter="\t")
        t = (e["timestep"] - e["timestep"][0]) * 1e-3
        axes[0].plot(t, e["potential_energy"] / 5000, color=cmap(norm(r.U0)), lw=0.8,
                     ls="--" if r.L_bridge == 80.0 else "-")
        win = e["timestep"] >= e["timestep"][-1] - SQ_WINDOW_STEPS
        axes[1].scatter(r.U0, e["temperature"][win].mean(), color=L_COLOR.get(r.L_bridge, INK2),
                        marker="o" if r.role == "train" else "x", s=22)
    t_end = (df.prod_steps.iloc[0]) * 1e-3
    axes[0].axvspan(t_end - SQ_WINDOW_STEPS * 1e-3, t_end, color=GRID, alpha=0.7, lw=0)
    axes[0].set_ylim(-48, 2)
    axes[0].text(t_end - 2500, -46.5, "S(q) window", ha="center", fontsize=8, color=INK2)
    axes[0].set_xlabel(r"production time ($\tau$)"); axes[0].set_ylabel("potential energy per particle (kT)")
    axes[0].set_title("(a) potential energy (solid L = 20, dashed L = 80; colour = U0)", loc="left", fontsize=9.5)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=axes[0], fraction=0.04, pad=0.01).set_label("U0 (kT)")
    axes[1].axhline(1.0, color=MUTED, lw=0.8, ls="--")
    axes[1].set_xlabel("U0 (kT)"); axes[1].set_ylabel("mean T in S(q) window")
    axes[1].set_ylim(0.995, 1.005)
    axes[1].set_title("(b) temperature", loc="left", fontsize=9.5)
    axes[1].legend([Line2D([], [], marker="o", color=BLUE, ls=""), Line2D([], [], marker="o", color=ORANGE, ls=""),
                    Line2D([], [], marker="x", color=INK2, ls="")],
                   ["L = 20", "L = 80", "held-out (L = 40, 120: grey)"], frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "r31_energy.png", bbox_inches="tight")
    plt.close(fig)


def fig_heldout(df, run, out):
    ho = df[df.role == "heldout"]
    tr = df[df.role == "train"]
    neighbours = {
        "h0": ["d6", "d18"],                 # same (C_anchor, NaCl), bridges 20 and 80
        "h1": ["d6", "d7"],                  # C_anchor 100 at 20 and 40 mM (nearest grid)
        "h2": ["d17", "d21"],                # L 80, C_anchor 100 / 140 at 10 mM
        "h3": ["d18"],                       # L 80 at the same (C_anchor, NaCl)
        "h4": ["d7"],                        # L 20, C_anchor 100, 40 mM
    }
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6), sharey=True)
    for ax, (_, r) in zip(axes, ho.iterrows()):
        sq = load_sq(run, r.id)
        ax.plot(sq[:, 0], sq[:, 1], color=INK, lw=1.8, label=f"{r.id} (held-out)")
        for cid, col in zip(neighbours[r.id], (BLUE, ORANGE)):
            t = tr[tr.id == cid].iloc[0]
            s2 = load_sq(run, cid)
            ax.plot(s2[:, 0], s2[:, 1], color=col, lw=1.1, alpha=0.9,
                    label=f"{cid}: L={t.L_bridge:g}, C_anchor={t.C_anchor:g}, NaCl={t.C_NaCl:g}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(1.5e-3, 0.135); ax.set_ylim(0.07, 40)
        ax.axhline(1, color=MUTED, lw=0.6)
        ax.set_title(f"{r.id}: L={r.L_bridge:g}, C_anchor={r.C_anchor:g}, NaCl={r.C_NaCl:g}\n{r.note}; U0={r.U0:.2f}, r0={r.r0:.2f}",
                     fontsize=8.5, loc="left")
        ax.set_xlabel(QA)
        ax.legend(frameon=False, fontsize=7, loc="upper right")
    axes[0].set_ylabel("S(q)")
    fig.tight_layout()
    fig.savefig(out / "r31_heldout_sq.png", bbox_inches="tight")
    plt.close(fig)


def _sq_panel(ax, run, r, color):
    sq = load_sq(run, r.id)
    ax.plot(sq[:, 0], sq[:, 1], color=color, lw=1.4, ls="--" if r.dispersed else "-")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1.5e-3, 0.135); ax.set_ylim(0.07, 40)
    ax.axhline(1, color=MUTED, lw=0.6)
    state = "dispersed" if r.dispersed else f"q1={r.q1:.4f}"
    ax.set_title(f"{r.id}: L={r.L_bridge:g}, C_anchor={r.C_anchor:g}, NaCl={r.C_NaCl:g}\n"
                 f"U0={r.U0:.2f}, r0={r.r0:.2f}; {state}", fontsize=8, loc="left")


def fig_sq_all_train(df, run, out):
    """d0-d23 in id order, 4 rows x 6 columns."""
    tr = df[df.role == "train"].copy()
    tr["k"] = tr.id.str[1:].astype(int)
    tr = tr.sort_values("k")
    fig, axes = plt.subplots(4, 6, figsize=(16, 10.5), sharex=True, sharey=True)
    for ax, (_, r) in zip(axes.flat, tr.iterrows()):
        _sq_panel(ax, run, r, L_COLOR[r.L_bridge])
    for ax in axes[-1, :]:
        ax.set_xlabel(QA)
    for ax in axes[:, 0]:
        ax.set_ylabel("S(q)")
    handles = [Line2D([], [], color=BLUE), Line2D([], [], color=ORANGE),
               Line2D([], [], color=INK2, ls="--")]
    fig.legend(handles, ["L_bridge = 20 bp", "L_bridge = 80 bp", "dashed: dispersed by the QC rule"],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.savefig(out / "r31_sq_train_d0_d23.png", bbox_inches="tight")
    plt.close(fig)


def fig_sq_all_heldout(df, run, out):
    """h0-h4, 1 row x 5 columns."""
    ho = df[df.role == "heldout"]
    colors = {20.0: BLUE, 40.0: AQUA, 80.0: ORANGE, 120.0: YELLOW}
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.4), sharex=True, sharey=True)
    for ax, (_, r) in zip(axes, ho.iterrows()):
        _sq_panel(ax, run, r, colors[r.L_bridge])
        ax.set_xlabel(QA)
        ax.text(0.97, 0.95, r.note, transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=INK2)
    axes[0].set_ylabel("S(q)")
    fig.tight_layout()
    fig.savefig(out / "r31_sq_heldout_h0_h4.png", bbox_inches="tight")
    plt.close(fig)


def main(run, summary, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(summary)
    fig_condition_map(df, out)
    fig_sq_grid(df, run, out)
    fig_transition(df, out)
    fig_energy(df, run, out)
    fig_heldout(df, run, out)
    fig_sq_all_train(df, run, out)
    fig_sq_all_heldout(df, run, out)
    print("figures written to", out)


if __name__ == "__main__":
    main(*sys.argv[1:4])

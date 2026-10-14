"""
R3.1 snapshot figure: the last GSD frame of six ground-truth conditions in one figure.

Usage: python snapshots_r31.py SNAP_DIR SUMMARY_CSV OUT_DIR

SNAP_DIR holds the gsd-tools renders snapshots/gt/<id>.png, made by
DNA_digitwin campaigns/r3.2-physics-validate/best_eval/plot_best_eval.py
(--gt ... --ids d4 d16 d8 d20 d11 d23 --view perspective --no-sq --out SNAP_DIR):
gsd-tools defaults except the perspective camera and the colour (red; black box).
Particles at the core radius sigma/2.
Positions are wrapped into the box, so a cluster crossing a face appears split.
(2026-10-13: renders moved from a local OVITO call to gsd-tools.)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PAIRS = [("d4", "d16"), ("d8", "d20"), ("d11", "d23")]   # same U0, L_bridge 20 / 80


def main(snap, summary, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(summary).set_index("id")
    fig, axes = plt.subplots(2, 3, figsize=(12, 9.2))
    for j, pair in enumerate(PAIRS):
        for i, cid in enumerate(pair):
            r = df.loc[cid]
            ax = axes[i, j]
            ax.imshow(plt.imread(Path(snap) / "snapshots" / "gt" / f"{cid}.png"))
            ax.set_axis_off()
            state = "dispersed" if r.dispersed else "peaked"
            ax.set_title(f"{cid}: L_bridge={r.L_bridge:g}, C_anchor={r.C_anchor:g}, NaCl={r.C_NaCl:g}\n"
                         f"U0={r.U0:.2f}, r0={r.r0:.2f}, n={r.n:.1f}, m={r.m:.1f}\n"
                         f"s_max-1 = {r.s_max_minus_1:.2f} ({state})", fontsize=8.5, loc="left")
    fig.text(0.01, 0.005, "Last frame (production t = 22,450 tau), gsd-tools perspective view. Red: particle cores; black: "
             "periodic box, L = 100 sigma; radius sigma/2. Columns: same U0; rows: L_bridge 20 (top) and 80 (bottom).",
             fontsize=8.5, color="#52514e")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out / "r31_snapshots.png", dpi=110, bbox_inches="tight")
    print("wrote", out / "r31_snapshots.png")


if __name__ == "__main__":
    main(*sys.argv[1:4])

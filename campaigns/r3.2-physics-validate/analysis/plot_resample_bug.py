"""Illustrate the endpoint-hold artifact of metrics._resample_reduced (truth evaluation, d3)."""
import sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root
import metrics
from metrics import shift_rmse_loss, _sanitize_curve
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixed_resample import resample_all_points, ORIG   # noqa: E402

GT = "/Users/hanson/Desktop/Sweep8_R3.1_GT/ground_truth"
TR = "/Users/hanson/Desktop/truth_eval/out_truth/eval_000"
cid = "d3"
gt = _sanitize_curve(np.load(f"{GT}/{cid}.npy"))
sim = _sanitize_curve(np.load(f"{TR}/{cid}/S(q)_data/average_structure_factor.npy"))
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, (fn, name) in zip(axes, ((ORIG, "repo `_resample_reduced` (in-window points only)"),
                                 (resample_all_points, "all points interpolated"))):
    metrics._resample_reduced = fn
    loss, d, grid, le, ls = shift_rmse_loss(gt, sim, None)
    ax.plot(gt[:, 0] / d["x_scale_tgt"], gt[:, 1], "o", color="#0b0b0b", ms=4, label="ground truth samples")
    ax.plot(sim[:, 0] / d["x_scale_sim"], sim[:, 1], "s", color="#eb6834", ms=4, mfc="none", label="truth-eval sim samples")
    ax.plot(grid, 10 ** np.asarray(le), color="#0b0b0b", lw=1.2, label="GT on log grid")
    ax.plot(grid, 10 ** np.asarray(ls), color="#eb6834", lw=1.6, label="sim on log grid")
    ax.axvline(d["x_lo"], color="#898781", ls="--", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.1, 0.45); ax.set_ylim(0.3, 25)
    ax.set_xlabel(r"$x = q/q_1$"); ax.set_title(f"{name}\nd3 loss = {loss:.4f} (M4 {d['m4']:.4f})", fontsize=9, loc="left")
    ax.grid(True, color="#e6e5e1", lw=0.6)
axes[0].set_ylabel("S(q)"); axes[0].legend(frameon=False, fontsize=7.5, loc="upper right")
fig.text(0.01, -0.03, "Dashed: lower end x_lo of the overlap. Left: the simulated curve's first sample lies below x_lo and is dropped, "
         "so the grid between x_lo and its second sample is held flat.", fontsize=8, color="#52514e")
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=150, bbox_inches="tight")
print("wrote", sys.argv[2])

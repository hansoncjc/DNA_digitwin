"""Compare metrics._resample_reduced (repo) with a variant that interpolates on all points.

The repo version builds each interpolator only from points inside [lo, hi] and holds the
first in-window value below it. The variant uses every point of each curve, so the value
at lo is interpolated from the bracketing samples. The grid and everything else are unchanged.
"""
import sys, json
import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root
import metrics
from metrics import shift_rmse_loss, _sanitize_curve, MetricFailed
from datasets import physics_map

ORIG = metrics._resample_reduced

def resample_all_points(exp_curve, sim_curve, window, exp_scale, sim_scale, n_points):
    qe = exp_curve[:, 0] / exp_scale; ie = exp_curve[:, 1]
    qs = sim_curve[:, 0] / sim_scale; isim = sim_curve[:, 1]
    lo = max(qe.min(), qs.min(), window[0]); hi = min(qe.max(), qs.max(), window[1])
    if not (hi > lo):
        raise MetricFailed("empty window")
    grid = np.logspace(np.log10(lo), np.log10(hi), int(n_points))
    i_exp = interp1d(qe, ie, kind="linear", bounds_error=False, fill_value=(ie[0], ie[-1]))(grid)
    i_sim = interp1d(qs, isim, kind="linear", bounds_error=False, fill_value=(isim[0], isim[-1]))(grid)
    return np.log10(np.clip(i_exp, 1e-10, None)), np.log10(np.clip(i_sim, 1e-10, None)), grid

GT = "/Users/hanson/Desktop/Sweep8_R3.1_GT/ground_truth"
TR = "/Users/hanson/Desktop/truth_eval/out_truth/eval_000"
cfg = json.load(open(f"{GT}/d0/gt_params.json"))["coefficients"]

def floor(variant):
    metrics._resample_reduced = variant
    rows = []
    for i in range(24):
        cid = f"d{i}"
        gt = _sanitize_curve(np.load(f"{GT}/{cid}.npy"))
        sim = _sanitize_curve(np.load(f"{TR}/{cid}/S(q)_data/average_structure_factor.npy"))
        rows.append(shift_rmse_loss(gt, sim, None)[0])
    return np.array(rows)

def kshift(variant, dk, way):
    metrics._resample_reduced = variant
    tot = 0.0
    for i in range(24):
        cid = f"d{i}"
        c = json.load(open(f"{GT}/{cid}/gt_params.json"))["condition"]
        r_gt = physics_map(c["L_bridge"], c["C_chol"], c["C_NaCl"], **cfg)["r0"]
        r_k = physics_map(c["L_bridge"], c["C_chol"], c["C_NaCl"], **{**cfg, "k": cfg["k"] + dk})["r0"]
        gt = _sanitize_curve(np.load(f"{GT}/{cid}.npy")); m = gt.copy()
        if way == 0: m[:, 0] = gt[:, 0] * r_gt / r_k
        else: m[:, 0] *= r_gt / r_k
        tot += shift_rmse_loss(gt, m, None)[0]
    return tot

f_orig, f_fix = floor(ORIG), floor(resample_all_points)
print("truth floor: repo %.3f   all-points interpolation %.3f" % (f_orig.sum(), f_fix.sum()))
for i, (a, b) in enumerate(zip(f_orig, f_fix)):
    print(f"  d{i:<3d} repo {a:.4f}  fixed {b:.4f}  diff {a-b:+.4f}")
print("\npure-shift k estimate (two numerically equivalent ways of building the shifted curve)")
for dk in (0.0, 0.005, 0.01, 0.02, 0.05):
    print(f"  dk={dk:<5} repo: {kshift(ORIG, dk, 0):.3f} / {kshift(ORIG, dk, 1):.3f}    fixed: {kshift(resample_all_points, dk, 0):.3f} / {kshift(resample_all_points, dk, 1):.3f}")

# share of the fixed-version floor below q = 0.006
metrics._resample_reduced = resample_all_points
sh = []
for i in range(24):
    cid = f"d{i}"
    gt = _sanitize_curve(np.load(f"{GT}/{cid}.npy"))
    sim = _sanitize_curve(np.load(f"{TR}/{cid}/S(q)_data/average_structure_factor.npy"))
    l, d, grid, le, ls = shift_rmse_loss(gt, sim, None)
    q = np.asarray(grid) * d["x_scale_tgt"]; d2 = (np.asarray(le) - np.asarray(ls)) ** 2
    sh.append(d2[q < 0.006].sum() / d2.sum())
print("\nfixed floor: share of squared error at q<0.006, median %.2f, range %.2f-%.2f" % (np.median(sh), min(sh), max(sh)))

"""
Check the ground-truth S(q) before training (no simulation).

    python qc_ground_truth.py --gt GT_ROOT [--csv GT_ROOT/qc.csv]

For each ``GT_ROOT/<id>.npy``: first-peak detection with the shift_rmse
defaults (q1, smoothed s_max - 1, dispersed or not) and a self-comparison
``shift_rmse_loss(gt, gt)``, which must give 0 and must not fail (asymptote
band covered, peak found). Curves with s_max - 1 within 0.25 of the
dispersed threshold (0.5) are flagged NEAR: a small change in a prediction
there switches between aligned and unaligned comparison.
Exit code 1 if any curve is missing or fails.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import gt_config as config  # noqa: E402
from metrics import SHIFT_RMSE_DEFAULTS, _sanitize_curve, detect_first_peak, shift_rmse_loss  # noqa: E402

NEAR = 0.25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="ground_truth")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    gt = Path(a.gt).resolve()
    if not gt.is_dir():
        sys.exit(f"ground-truth folder not found: {gt} (run from the GT folder or pass --gt)")
    print(f"ground truth: {gt}")
    delta = SHIFT_RMSE_DEFAULTS["dispersed_delta"]

    rows, bad = [], 0
    for c in config.ALL_CONDITIONS:
        path = gt / f"{c['id']}.npy"
        row = {"id": c["id"], "role": c["role"], "L_bridge": c["L_bridge"],
               "C_chol": c["C_chol"], "C_NaCl": c["C_NaCl"]}
        if not path.exists():
            row["status"] = "MISSING"
            bad += 1
            rows.append(row)
            continue
        curve = _sanitize_curve(np.load(path))
        peak = detect_first_peak(curve)
        row.update(n_points=len(curve), q_max=float(curve[-1, 0]),
                   s_max_minus_1=peak.s_max - 1.0, dispersed=bool(peak.dispersed),
                   q1=peak.q1)
        try:
            self_loss = float(shift_rmse_loss(curve, curve, None)[0])
            row["self_loss"] = self_loss
            row["status"] = "OK" if abs(self_loss) < 1e-9 else "SELF_LOSS_NONZERO"
        except Exception as e:  # MetricFailed or shape problems
            row["status"] = f"FAIL: {e}"
        if row["status"] != "OK":
            bad += 1
        if abs(row["s_max_minus_1"] - delta) < NEAR:
            row["status"] += " NEAR"
        rows.append(row)

    keys = ["id", "role", "L_bridge", "C_chol", "C_NaCl", "n_points", "q_max",
            "s_max_minus_1", "dispersed", "q1", "self_loss", "status"]
    print(" ".join(f"{k:>12}" for k in keys))
    for r in rows:
        print(" ".join(f"{(f'{r[k]:.4g}' if isinstance(r.get(k), float) else str(r.get(k, ''))):>12}"
                       for k in keys))
    n_disp = sum(1 for r in rows if r.get("dispersed"))
    print(f"\n{len(rows)} conditions, {n_disp} dispersed, {bad} missing/failed")
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows({k: r.get(k, "") for k in keys} for r in rows)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

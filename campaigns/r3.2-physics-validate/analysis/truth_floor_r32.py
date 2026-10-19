"""
Truth-loss floor of R3.2 from the two truth evaluations (ground-truth
coefficients, seed 42 against the ground truth's seed 1), per condition.

Usage: python truth_floor_r32.py GT_DIR V1_DIR V2_DIAG_DIR

GT_DIR: Sweep8_R3.1_GT (holds ground_truth/<id>.npy). V1_DIR: the first truth
evaluation's eval_000/ (run before the shift_rmse resampling fix 0ec9183; its
S(q) are rescored here with the repo shift_rmse_loss). V2_DIAG_DIR: the
shift_rmse_diagnostics.json of truth evaluation v2 (job 41696968, code
2897707), one folder per condition. All read only.

Writes r32_truth_floor.csv next to this script: loss_v1, loss_v2 (one column
per truth evaluation, read by plot_r32.py --floor-csv), the spacing terms, the
simulated and target smoothed s_max - 1, and the share of v1's squared error
at q < 0.006 1/A.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root
from metrics import _sanitize_curve, shift_rmse_loss  # noqa: E402

LOW_Q = 0.006


def main(gt_dir, v1_dir, v2_diag):
    rows = []
    for i in range(24):
        cid = f"d{i}"
        g = _sanitize_curve(np.load(Path(gt_dir) / "ground_truth" / f"{cid}.npy"))
        s = _sanitize_curve(np.load(Path(v1_dir) / cid / "S(q)_data" / "average_structure_factor.npy"))
        loss, d, grid, le, ls = shift_rmse_loss(g, s, None)
        q = np.asarray(grid) * d["x_scale_tgt"]
        d2 = (np.asarray(le) - np.asarray(ls)) ** 2
        v2 = json.loads((Path(v2_diag) / cid / "shift_rmse_diagnostics.json").read_text())
        rows.append({
            "id": cid, "loss_v1": loss, "loss_v2": v2["loss"],
            "shift_v1": d["shift_term"], "shift_v2": v2["shift_term"],
            "smax1_tgt": v2["peak_tgt"]["s_max"] - 1,
            "smax1_v1": d["peak_sim"]["s_max"] - 1, "smax1_v2": v2["peak_sim"]["s_max"] - 1,
            "lowq_share_v1": d2[q < LOW_Q].sum() / d2.sum() if d2.sum() > 0 else np.nan,
        })
    t = pd.DataFrame(rows)
    t.to_csv(Path(__file__).resolve().parent / "r32_truth_floor.csv", index=False, float_format="%.6g")
    print(f"total v1 {t.loss_v1.sum():.4f}  v2 {t.loss_v2.sum():.4f}  "
          f"difference {t.loss_v2.sum() - t.loss_v1.sum():+.4f}")


if __name__ == "__main__":
    main(*sys.argv[1:4])

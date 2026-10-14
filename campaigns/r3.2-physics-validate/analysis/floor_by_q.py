"""Where the truth-loss floor comes from: share of the M4 mean-squared log difference by q band."""
import sys, json
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root
from metrics import shift_rmse_loss, _sanitize_curve
GT = "/Users/hanson/Desktop/Sweep8_R3.1_GT/ground_truth"
TR = "/Users/hanson/Desktop/truth_eval/out_truth/eval_000"
bands = [(0, 0.006), (0.006, 0.03), (0.03, 1)]
print("id    loss_repo  loss_recomp  share q<0.006  0.006-0.03  >0.03")
rows = []
for i in range(24):
    cid = f"d{i}"
    gt = _sanitize_curve(np.load(f"{GT}/{cid}.npy"))
    sim = _sanitize_curve(np.load(f"{TR}/{cid}/S(q)_data/average_structure_factor.npy"))
    loss, diag, grid, le, ls = shift_rmse_loss(gt, sim, None)
    q = np.asarray(grid) * diag["x_scale_tgt"]
    d2 = (np.asarray(le) - np.asarray(ls)) ** 2
    tot = d2.sum()
    shares = [d2[(q >= a) & (q < b)].sum() / tot for a, b in bands]
    rep = json.load(open(f"{TR}/{cid}/shift_rmse_diagnostics.json"))["loss"]
    rows.append((cid, rep, loss, *shares))
    print(f"{cid:4s} {rep:10.5f} {loss:11.5f}   {shares[0]:12.2f} {shares[1]:11.2f} {shares[2]:6.2f}")
